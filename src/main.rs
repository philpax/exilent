use std::{collections::HashMap, env, io::Cursor, time::Duration};

use anyhow::Context as AnyhowContext;
use dotenv::dotenv;
use serenity::{
    async_trait,
    client::{Context, EventHandler},
    http::Http,
    model::{
        application::interaction::{Interaction, InteractionResponseType},
        prelude::{
            command::{Command, CommandOptionType},
            component::InputTextStyle,
            interaction::{
                application_command::ApplicationCommandInteraction,
                message_component::MessageComponentInteraction, modal::ModalSubmitInteraction,
            },
            *,
        },
    },
    Client,
};

mod consts;
mod custom_id;
mod store;
mod util;

use custom_id as cid;
use stable_diffusion_a1111_webui_client as sd;
use store::Store;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();

    let client = {
        let sd_url = env::var("SD_URL").expect("SD_URL not specified");
        let sd_authentication = env::var("SD_USER").ok().zip(env::var("SD_PASS").ok());
        sd::Client::new(
            &sd_url,
            sd_authentication
                .as_ref()
                .map(|p| sd::Authentication::ApiAuth(&p.0, &p.1))
                .unwrap_or(sd::Authentication::None),
        )
        .await
        .unwrap()
    };
    let models = client.models().await?;
    let store = Store::load()?;

    // Build our client.
    let mut client = Client::builder(
        env::var("DISCORD_TOKEN").expect("Expected a token in the environment"),
        GatewayIntents::default(),
    )
    .event_handler(Handler {
        client,
        models,
        store,
    })
    .await
    .expect("Error creating client");

    // Finally, start a single shard, and start listening to events.
    // Shards will automatically attempt to reconnect, and will perform
    // exponential backoff until it reconnects.
    if let Err(why) = client.start().await {
        println!("Client error: {:?}", why);
    }

    Ok(())
}

struct Handler {
    client: sd::Client,
    models: Vec<sd::Model>,
    store: Store,
}
/// Commands
impl Handler {
    async fn exilent(&self, http: &Http, cmd: ApplicationCommandInteraction) -> anyhow::Result<()> {
        let channel = cmd.channel_id;
        let texts = match self.client.embeddings().await {
            Ok(embeddings) => {
                util::generate_chunked_strings(embeddings.iter().map(|s| format!("`{s}`")), 1900)
            }
            Err(err) => vec![format!("{err:?}")],
        };
        cmd.create_interaction_response(http, |response| {
            response
                .kind(InteractionResponseType::ChannelMessageWithSource)
                .interaction_response_data(|message| {
                    message.title("Embeddings").content(texts.first().unwrap())
                })
        })
        .await?;

        for remainder in texts.iter().skip(1) {
            channel
                .send_message(http, |msg| msg.content(remainder))
                .await?;
        }

        Ok(())
    }

    async fn paint(&self, http: &Http, aci: ApplicationCommandInteraction) -> anyhow::Result<()> {
        let interaction: &dyn DiscordInteraction = &aci;

        // always applied
        let prompt = util::get_value(&aci, consts::value::PROMPT)
            .and_then(util::value_to_string)
            .context("expected prompt")?;

        let negative_prompt =
            util::get_value(&aci, consts::value::NEGATIVE_PROMPT).and_then(util::value_to_string);

        let seed = util::get_value(&aci, consts::value::SEED).and_then(util::value_to_int);

        let batch_count = util::get_value(&aci, consts::value::COUNT)
            .and_then(util::value_to_int)
            .map(|v| v as u32);

        // use last generation as default if available
        let last_generation = self.store.get_last_generation_for_user(aci.user.id)?;
        let last_generation = last_generation.as_ref();

        let width = util::get_value(&aci, consts::value::WIDTH)
            .and_then(util::value_to_int)
            .map(|v| v as u32 / 64 * 64)
            .or_else(|| last_generation.map(|g| g.width));

        let height = util::get_value(&aci, consts::value::HEIGHT)
            .and_then(util::value_to_int)
            .map(|v| v as u32 / 64 * 64)
            .or_else(|| last_generation.map(|g| g.height));

        let cfg_scale = util::get_value(&aci, consts::value::GUIDANCE_SCALE)
            .and_then(util::value_to_number)
            .map(|v| v as f32)
            .or_else(|| last_generation.map(|g| g.cfg_scale));

        let steps = util::get_value(&aci, consts::value::STEPS)
            .and_then(util::value_to_int)
            .map(|v| v as u32)
            .or_else(|| last_generation.map(|g| g.steps));

        let tiling = util::get_value(&aci, consts::value::TILING)
            .and_then(util::value_to_bool)
            .or_else(|| last_generation.map(|g| g.tiling));

        let restore_faces = util::get_value(&aci, consts::value::RESTORE_FACES)
            .and_then(util::value_to_bool)
            .or_else(|| last_generation.map(|g| g.restore_faces));

        let sampler = util::get_value(&aci, consts::value::SAMPLER)
            .and_then(util::value_to_string)
            .and_then(|v| sd::Sampler::try_from(v.as_str()).ok())
            .or_else(|| last_generation.map(|g| g.sampler));

        let model = {
            let models: Vec<_> = util::get_values_starting_with(&aci, consts::value::MODEL)
                .flat_map(util::value_to_string)
                .collect();
            if models.len() > 1 {
                anyhow::bail!("more than one model specified: {:?}", models);
            }

            models
                .first()
                .and_then(|v| self.models.iter().find(|m| m.title == *v))
                .or_else(|| {
                    last_generation.and_then(|g| {
                        util::find_model_by_hash(&self.models, &g.model_hash).map(|t| t.1)
                    })
                })
        };

        interaction
            .create(
                http,
                &format!(
                    "`{}`{}: Generating...",
                    &prompt,
                    negative_prompt
                        .as_deref()
                        .map(|s| format!(" - `{s}`"))
                        .unwrap_or_default()
                ),
            )
            .await?;

        issue_generation_task(
            &self.client,
            &self.models,
            &self.store,
            http,
            interaction,
            &sd::GenerationRequest {
                prompt: prompt.as_str(),
                negative_prompt: negative_prompt.as_deref(),
                seed,
                batch_size: Some(1),
                batch_count,
                width,
                height,
                cfg_scale,
                steps,
                tiling,
                restore_faces,
                sampler,
                model,
                ..Default::default()
            },
        )
        .await
    }

    async fn interrogate(
        &self,
        http: &Http,
        aci: ApplicationCommandInteraction,
    ) -> anyhow::Result<()> {
        let image_url = util::get_value(&aci, consts::value::IMAGE_URL)
            .and_then(util::value_to_string)
            .context("expected image_url")?;

        let interrogator = util::get_value(&aci, consts::value::INTERROGATOR)
            .and_then(util::value_to_string)
            .and_then(|v| sd::Interrogator::try_from(v.as_str()).ok())
            .context("expected interrogator")?;

        let interaction: &dyn DiscordInteraction = &aci;
        interaction
            .create(
                http,
                &format!("Interrogating {image_url} with {interrogator}..."),
            )
            .await?;

        let bytes = reqwest::get(&image_url).await?.bytes().await?;
        let image = image::load_from_memory(&bytes)?;

        issue_interrogate_task(
            &self.client,
            interaction,
            &self.store,
            http,
            image,
            store::InterrogationSource::Url(image_url),
            interrogator,
        )
        .await
    }
}
/// Message component interactions
impl Handler {
    async fn mc_retry(
        &self,
        http: &Http,
        mci: &MessageComponentInteraction,
        id: i64,
    ) -> anyhow::Result<()> {
        self.mc_retry_impl(http, mci, id, None, None, Some(None))
            .await
    }

    async fn mc_retry_with_options(
        &self,
        http: &Http,
        mci: &MessageComponentInteraction,
        id: i64,
    ) -> anyhow::Result<()> {
        let (old_prompt, negative_prompt, seed) = self
            .store
            .get_generation(id)?
            .map(|g| (g.prompt, g.negative_prompt, g.seed))
            .context("generation not found")?;

        mci.create_interaction_response(http, |r| {
            r.kind(InteractionResponseType::Modal)
                .interaction_response_data(|d| {
                    d.components(|c| {
                        c.create_action_row(|r| {
                            r.create_input_text(|t| {
                                t.label("Prompt")
                                    .custom_id(consts::value::PROMPT)
                                    .required(true)
                                    .style(InputTextStyle::Short)
                                    .value(old_prompt)
                            })
                        })
                        .create_action_row(|r| {
                            r.create_input_text(|t| {
                                t.label("Negative prompt")
                                    .custom_id(consts::value::NEGATIVE_PROMPT)
                                    .required(false)
                                    .style(InputTextStyle::Short)
                                    .value(negative_prompt.unwrap_or_default())
                            })
                        })
                        .create_action_row(|r| {
                            r.create_input_text(|t| {
                                t.label("Seed")
                                    .custom_id(consts::value::SEED)
                                    .required(false)
                                    .style(InputTextStyle::Short)
                                    .value(seed)
                            })
                        })
                    })
                    .title("Retry with prompt")
                    .custom_id(cid::Generation::RetryWithOptionsResponse.to_id(id))
                })
        })
        .await?;

        Ok(())
    }

    async fn mc_retry_with_options_response(
        &self,
        http: &Http,
        msi: &ModalSubmitInteraction,
        id: i64,
    ) -> anyhow::Result<()> {
        let rows: HashMap<String, String> = msi
            .data
            .components
            .iter()
            .flat_map(|r| r.components.iter())
            .filter_map(|c| {
                if let component::ActionRowComponent::InputText(it) = c {
                    Some((it.custom_id.clone(), it.value.clone()))
                } else {
                    None
                }
            })
            .collect();

        let prompt = rows.get(consts::value::PROMPT).map(|s| s.as_str());
        let negative_prompt = rows.get(consts::value::NEGATIVE_PROMPT).map(|s| s.as_str());
        let seed = rows.get(consts::value::SEED).map(|s| s.parse::<i64>().ok());

        self.mc_retry_impl(http, msi, id, prompt, negative_prompt, seed)
            .await
    }

    async fn mc_retry_impl(
        &self,
        http: &Http,
        interaction: &dyn DiscordInteraction,
        id: i64,
        prompt: Option<&str>,
        negative_prompt: Option<&str>,
        // None: don't override; Some(None): override with a fresh seed; Some(Some(seed)): override with seed
        seed: Option<Option<i64>>,
    ) -> anyhow::Result<()> {
        let generation = self
            .store
            .get_generation(id)?
            .context("generation not found")?
            .clone();

        let mut request = generation.as_generation_request(&self.models);
        if let Some(prompt) = prompt {
            request.prompt = prompt;
        }
        if let Some(negative_prompt) = negative_prompt {
            request.negative_prompt = Some(negative_prompt).filter(|s| !s.is_empty());
        }
        if let Some(seed) = seed {
            request.seed = seed;
        }
        interaction
            .create(
                http,
                &format!(
                    "`{}`{}: Generating retry...",
                    request.prompt,
                    request
                        .negative_prompt
                        .map(|s| format!(" - `{s}`"))
                        .unwrap_or_default()
                ),
            )
            .await?;

        issue_generation_task(
            &self.client,
            &self.models,
            &self.store,
            http,
            interaction,
            &request,
        )
        .await?;

        Ok(())
    }

    async fn mc_interrogate(
        &self,
        http: &Http,
        interaction: &dyn DiscordInteraction,
        id: i64,
        interrogator: sd::Interrogator,
    ) -> anyhow::Result<()> {
        interaction.create(http, "Interrogating...").await?;

        let image = image::load_from_memory(
            &self
                .store
                .get_generation(id)?
                .context("generation not found")?
                .image,
        )?;

        issue_interrogate_task(
            &self.client,
            interaction,
            &self.store,
            http,
            image,
            store::InterrogationSource::GenerationId(id),
            interrogator,
        )
        .await
    }

    async fn mc_interrogate_generate(
        &self,
        http: &Http,
        interaction: &dyn DiscordInteraction,
        id: i64,
    ) -> anyhow::Result<()> {
        let interrogation = self
            .store
            .get_interrogation(id)?
            .context("no interrogation found")?;

        // always applied
        let prompt = interrogation.result;

        // use last generation as default if available
        let last_generation = self
            .store
            .get_last_generation_for_user(interaction.user().id)?;
        let last_generation = last_generation.as_ref();

        let width = last_generation.map(|g| g.width);
        let height = last_generation.map(|g| g.height);
        let cfg_scale = last_generation.map(|g| g.cfg_scale);
        let steps = last_generation.map(|g| g.steps);
        let tiling = last_generation.map(|g| g.tiling);
        let restore_faces = last_generation.map(|g| g.restore_faces);
        let sampler = last_generation.map(|g| g.sampler);
        let model = last_generation
            .and_then(|g| util::find_model_by_hash(&self.models, &g.model_hash).map(|t| t.1));

        interaction
            .create(http, &format!("`{prompt}`: Generating..."))
            .await?;

        issue_generation_task(
            &self.client,
            &self.models,
            &self.store,
            http,
            interaction,
            &sd::GenerationRequest {
                prompt: prompt.as_str(),
                negative_prompt: None,
                seed: None,
                batch_size: Some(1),
                batch_count: Some(1),
                width,
                height,
                cfg_scale,
                steps,
                tiling,
                restore_faces,
                sampler,
                model,
                ..Default::default()
            },
        )
        .await
    }
}

#[async_trait]
impl EventHandler for Handler {
    async fn ready(&self, ctx: Context, ready: Ready) {
        println!("{} is connected!", ready.user.name);

        Command::create_global_application_command(&ctx.http, |command| {
            command
                .name(consts::command::PAINT)
                .description("Paints your dreams")
                .create_option(|option| {
                    option
                        .name(consts::value::PROMPT)
                        .description("The prompt to draw")
                        .kind(CommandOptionType::String)
                        .required(true)
                })
                .create_option(|option| {
                    option
                        .name(consts::value::NEGATIVE_PROMPT)
                        .description("The prompt to avoid drawing")
                        .kind(CommandOptionType::String)
                        .required(false)
                })
                .create_option(|option| {
                    option
                        .name(consts::value::SEED)
                        .description("The seed to use")
                        .kind(CommandOptionType::Integer)
                        .required(false)
                })
                .create_option(|option| {
                    option
                        .name(consts::value::COUNT)
                        .description("The number of images to generate")
                        .kind(CommandOptionType::Integer)
                        .min_int_value(consts::limits::COUNT_MIN)
                        .max_int_value(consts::limits::COUNT_MAX)
                        .required(false)
                })
                .create_option(|option| {
                    option
                        .name(consts::value::WIDTH)
                        .description("The width of the image")
                        .kind(CommandOptionType::Integer)
                        .min_int_value(consts::limits::WIDTH_MIN)
                        .max_int_value(consts::limits::WIDTH_MAX)
                        .required(false)
                })
                .create_option(|option| {
                    option
                        .name(consts::value::HEIGHT)
                        .description("The height of the image")
                        .kind(CommandOptionType::Integer)
                        .min_int_value(consts::limits::HEIGHT_MIN)
                        .max_int_value(consts::limits::HEIGHT_MAX)
                        .required(false)
                })
                .create_option(|option| {
                    option
                        .name(consts::value::GUIDANCE_SCALE)
                        .description("The scale of the guidance to apply")
                        .kind(CommandOptionType::Number)
                        .min_number_value(consts::limits::GUIDANCE_SCALE_MIN)
                        .max_number_value(consts::limits::GUIDANCE_SCALE_MAX)
                        .required(false)
                })
                .create_option(|option| {
                    option
                        .name(consts::value::STEPS)
                        .description("The number of denoising steps to apply")
                        .kind(CommandOptionType::Integer)
                        .min_int_value(consts::limits::STEPS_MIN)
                        .max_int_value(consts::limits::STEPS_MAX)
                        .required(false)
                })
                .create_option(|option| {
                    option
                        .name(consts::value::TILING)
                        .description("Whether or not the image should be tiled at the edges")
                        .kind(CommandOptionType::Boolean)
                        .required(false)
                })
                .create_option(|option| {
                    option
                        .name(consts::value::RESTORE_FACES)
                        .description("Whether or not the image should have its faces restored")
                        .kind(CommandOptionType::Boolean)
                        .required(false)
                })
                .create_option(|option| {
                    let opt = option
                        .name(consts::value::SAMPLER)
                        .description("The sampler to use")
                        .kind(CommandOptionType::String)
                        .required(false);

                    for value in sd::Sampler::VALUES {
                        opt.add_string_choice(value, value);
                    }

                    opt
                });

            for (idx, chunk) in self
                .models
                .chunks(consts::misc::MODEL_CHUNK_COUNT)
                .enumerate()
            {
                command.create_option(|option| {
                    let opt = option
                        .name(if idx == 0 {
                            consts::value::MODEL.to_string()
                        } else {
                            format!("{}{}", consts::value::MODEL, idx + 1)
                        })
                        .description(format!("The model to use, category {}", idx + 1))
                        .kind(CommandOptionType::String)
                        .required(false);

                    for model in chunk {
                        opt.add_string_choice(&model.name, &model.title);
                    }

                    opt
                });
            }

            command
        })
        .await
        .unwrap();

        Command::create_global_application_command(&ctx.http, |command| {
            command
                .name(consts::command::INTERROGATE)
                .description("Interrogates an image to produce a caption")
                .create_option(|option| {
                    option
                        .name(consts::value::IMAGE_URL)
                        .description("The URL of the image to interrogate")
                        .kind(CommandOptionType::String)
                        .required(true)
                })
                .create_option(|option| {
                    let opt = option
                        .name(consts::value::INTERROGATOR)
                        .description("The interrogator to use")
                        .kind(CommandOptionType::String)
                        .required(true);

                    for value in sd::Interrogator::VALUES {
                        opt.add_string_choice(value, value);
                    }

                    opt
                })
        })
        .await
        .unwrap();

        Command::create_global_application_command(&ctx.http, |command| {
            command
                .name(consts::command::EXILENT)
                .description("Meta-commands for Exilent")
                .create_option(|option| {
                    option
                        .name("embeddings")
                        .description("Lists all of the supported embeddings")
                        .kind(CommandOptionType::SubCommand)
                })
        })
        .await
        .unwrap();
    }

    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        let mut channel_id = None;
        let http = &ctx.http;
        let result = match interaction {
            Interaction::ApplicationCommand(cmd) => {
                channel_id = Some(cmd.channel_id);
                match cmd.data.name.as_str() {
                    consts::command::PAINT => self.paint(http, cmd).await,
                    consts::command::INTERROGATE => self.interrogate(http, cmd).await,
                    consts::command::EXILENT => self.exilent(http, cmd).await,
                    _ => Ok(()),
                }
            }
            Interaction::MessageComponent(mci) => {
                channel_id = Some(mci.channel_id);

                let custom_id = cid::CustomId::try_from(mci.data.custom_id.as_str())
                    .expect("invalid interaction id");

                match custom_id {
                    cid::CustomId::Generation { id, generation } => match generation {
                        cid::Generation::Retry => self.mc_retry(http, &mci, id).await,
                        cid::Generation::RetryWithOptions => {
                            self.mc_retry_with_options(http, &mci, id).await
                        }
                        cid::Generation::InterrogateClip => {
                            self.mc_interrogate(http, &mci, id, sd::Interrogator::Clip)
                                .await
                        }
                        cid::Generation::InterrogateDeepDanbooru => {
                            self.mc_interrogate(http, &mci, id, sd::Interrogator::DeepDanbooru)
                                .await
                        }
                        cid::Generation::RetryWithOptionsResponse => Ok(()),
                    },
                    cid::CustomId::Interrogation { id, interrogation } => match interrogation {
                        cid::Interrogation::Generate => {
                            self.mc_interrogate_generate(http, &mci, id).await
                        }
                    },
                }
            }
            Interaction::ModalSubmit(msi) => {
                channel_id = Some(msi.channel_id);

                let custom_id = cid::CustomId::try_from(msi.data.custom_id.as_str())
                    .expect("invalid interaction id");

                match custom_id {
                    cid::CustomId::Generation { id, generation } => match generation {
                        cid::Generation::RetryWithOptionsResponse => {
                            self.mc_retry_with_options_response(http, &msi, id).await
                        }

                        cid::Generation::Retry => Ok(()),
                        cid::Generation::RetryWithOptions => Ok(()),
                        cid::Generation::InterrogateClip => Ok(()),
                        cid::Generation::InterrogateDeepDanbooru => Ok(()),
                    },
                    cid::CustomId::Interrogation { .. } => todo!(),
                }
            }
            _ => Ok(()),
        };
        if let Some(channel_id) = channel_id {
            if let Err(err) = result.as_ref() {
                channel_id
                    .send_message(http, |r| r.content(format!("Error: {err:?}")))
                    .await
                    .ok();

                result.unwrap();
            }
        }
    }
}

async fn issue_generation_task(
    client: &sd::Client,
    models: &[sd::Model],
    store: &Store,
    http: &Http,
    interaction: &dyn DiscordInteraction,
    request: &sd::GenerationRequest<'_>,
) -> anyhow::Result<()> {
    use consts::misc::{PROGRESS_SCALE_FACTOR, PROGRESS_UPDATE_MS};

    let prompt = request.prompt;
    let negative_prompt = request.negative_prompt;

    // generate and update progress
    let task = client.generate_image_from_text(request)?;
    loop {
        let progress = task.progress().await?;
        let image_bytes = progress
            .current_image
            .as_ref()
            .map(|i| {
                encode_image_for_attachment(i.resize(
                    i.width() / PROGRESS_SCALE_FACTOR,
                    i.height() / PROGRESS_SCALE_FACTOR,
                    image::imageops::FilterType::Nearest,
                ))
            })
            .transpose()?;

        interaction
            .get_interaction_message(http)
            .await?
            .edit(http, |m| {
                m.content(format!(
                    "`{}`{}: {:.02}% complete. ({:.02} seconds remaining)",
                    request.prompt,
                    request
                        .negative_prompt
                        .map(|s| format!(" - `{s}`"))
                        .unwrap_or_default(),
                    progress.progress_factor * 100.0,
                    progress.eta_seconds
                ));

                if let Some(image_bytes) = &image_bytes {
                    if let Some(a) = m.0.get_mut("attachments").and_then(|e| e.as_array_mut()) {
                        a.clear();
                    }
                    m.attachment((image_bytes.as_slice(), "progress.png"));
                }

                m
            })
            .await?;

        if progress.is_finished() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(PROGRESS_UPDATE_MS)).await;
    }

    // retrieve result
    let result = task.block().await?;
    let images = result
        .images
        .into_iter()
        .enumerate()
        .map(|(idx, image)| {
            Ok((
                format!("image_{idx}.png"),
                encode_image_for_attachment(image)?,
            ))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // send images
    for (idx, ((filename, bytes), seed)) in images.iter().zip(result.info.seeds.iter()).enumerate()
    {
        interaction
            .get_interaction_message(http)
            .await?
            .edit(http, |m| {
                m.content(format!(
                    "`{}`{}: Uploading {}/{}...",
                    request.prompt,
                    request
                        .negative_prompt
                        .map(|s| format!(" - `{s}`"))
                        .unwrap_or_default(),
                    idx + 1,
                    images.len()
                ))
            })
            .await?;

        let generation = store::Generation {
            prompt: prompt.to_owned(),
            seed: *seed,
            width: result.info.width,
            height: result.info.height,
            cfg_scale: result.info.cfg_scale,
            steps: result.info.steps,
            tiling: result.info.tiling,
            restore_faces: result.info.restore_faces,
            sampler: result.info.sampler,
            negative_prompt: negative_prompt
                .map(|s| s.to_string())
                .filter(|p| !p.is_empty()),
            model_hash: result.info.model_hash.clone(),
            image: bytes.clone(),
            timestamp: result.info.job_timestamp,
            user_id: interaction.user().id,
        };
        let message = format!(
            "{} - {}",
            generation.as_message(models),
            interaction.user().mention()
        );
        let store_key = store.insert_generation(generation)?;

        use consts::emojis as E;
        interaction
            .channel_id()
            .send_files(&http, [(bytes.as_slice(), filename.as_str())], |m| {
                m.content(message).components(|c| {
                    c.create_action_row(|r| {
                        r.create_button(|b| {
                            b.emoji(E::RETRY.parse::<ReactionType>().unwrap())
                                .label("Retry")
                                .style(component::ButtonStyle::Secondary)
                                .custom_id(cid::Generation::Retry.to_id(store_key))
                        })
                        .create_button(|b| {
                            b.emoji(E::RETRY_WITH_OPTIONS.parse::<ReactionType>().unwrap())
                                .label("Retry with options")
                                .style(component::ButtonStyle::Secondary)
                                .custom_id(cid::Generation::RetryWithOptions.to_id(store_key))
                        })
                    })
                    .create_action_row(|r| {
                        r.create_button(|b| {
                            b.emoji(E::INTERROGATE_WITH_CLIP.parse::<ReactionType>().unwrap())
                                .label("CLIP")
                                .style(component::ButtonStyle::Secondary)
                                .custom_id(cid::Generation::InterrogateClip.to_id(store_key))
                        })
                        .create_button(|b| {
                            b.emoji(
                                E::INTERROGATE_WITH_DEEPDANBOORU
                                    .parse::<ReactionType>()
                                    .unwrap(),
                            )
                            .label("DeepDanbooru")
                            .style(component::ButtonStyle::Secondary)
                            .custom_id(cid::Generation::InterrogateDeepDanbooru.to_id(store_key))
                        })
                    })
                });

                if let Some(message) = interaction.message() {
                    m.reference_message(message);
                }

                m
            })
            .await?;
    }
    interaction
        .get_interaction_message(http)
        .await?
        .delete(http)
        .await?;

    Ok(())
}

async fn issue_interrogate_task(
    client: &sd::Client,
    interaction: &dyn DiscordInteraction,
    store: &Store,
    http: &Http,
    image: image::DynamicImage,
    source: store::InterrogationSource,
    interrogator: sd::Interrogator,
) -> Result<(), anyhow::Error> {
    let result = client.interrogate(&image, interrogator).await?;
    let store_key = store.insert_interrogation(store::Interrogation {
        user_id: interaction.user().id,
        source: source.clone(),
        result: result.clone(),
        interrogator,
    })?;

    interaction
        .get_interaction_message(http)
        .await?
        .edit(http, |m| {
            m.content(format!(
                "`{}` - {}{} for {}",
                result,
                interrogator,
                match source {
                    store::InterrogationSource::GenerationId(_) => String::new(),
                    store::InterrogationSource::Url(url) => format!(" on {url}"),
                },
                interaction.user().mention()
            ))
            .components(|c| {
                c.create_action_row(|r| {
                    r.create_button(|b| {
                        b.emoji(
                            consts::emojis::INTERROGATE_GENERATE
                                .parse::<ReactionType>()
                                .unwrap(),
                        )
                        .label("Generate")
                        .style(component::ButtonStyle::Secondary)
                        .custom_id(cid::Interrogation::Generate.to_id(store_key))
                    })
                })
            })
        })
        .await?;

    Ok(())
}

fn encode_image_for_attachment(image: image::DynamicImage) -> anyhow::Result<Vec<u8>> {
    let mut bytes: Vec<u8> = Vec::new();
    let mut cursor = Cursor::new(&mut bytes);
    image.write_to(&mut cursor, image::ImageOutputFormat::Png)?;
    Ok(bytes)
}

#[async_trait]
trait DiscordInteraction: Send + Sync {
    async fn create(&self, http: &Http, message: &str) -> anyhow::Result<()>;
    async fn get_interaction_message(&self, http: &Http) -> anyhow::Result<Message>;

    fn channel_id(&self) -> ChannelId;
    fn message(&self) -> Option<&Message>;
    fn user(&self) -> &User;
}
macro_rules! implement_interaction {
    ($name:ident) => {
        #[async_trait]
        impl DiscordInteraction for $name {
            async fn create(&self, http: &Http, msg: &str) -> anyhow::Result<()> {
                Ok(self
                    .create_interaction_response(http, |response| {
                        response
                            .kind(InteractionResponseType::ChannelMessageWithSource)
                            .interaction_response_data(|message| message.content(msg))
                    })
                    .await?)
            }
            async fn get_interaction_message(&self, http: &Http) -> anyhow::Result<Message> {
                Ok(self.get_interaction_response(http).await?)
            }

            fn channel_id(&self) -> ChannelId {
                self.channel_id
            }
            fn user(&self) -> &User {
                &self.user
            }
            interaction_message!($name);
        }
    };
}
macro_rules! interaction_message {
    (ApplicationCommandInteraction) => {
        fn message(&self) -> Option<&Message> {
            None
        }
    };
    (MessageComponentInteraction) => {
        fn message(&self) -> Option<&Message> {
            Some(&self.message)
        }
    };
    (ModalSubmitInteraction) => {
        fn message(&self) -> Option<&Message> {
            self.message.as_ref()
        }
    };
}
implement_interaction!(ApplicationCommandInteraction);
implement_interaction!(MessageComponentInteraction);
implement_interaction!(ModalSubmitInteraction);
