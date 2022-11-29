use std::{
    collections::HashMap,
    env,
    io::Cursor,
    sync::{Arc, Mutex},
    time::Duration,
};

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

use stable_diffusion_a1111_webui_client as sd;
use store::Store;

mod store;
mod util;

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
    let store = Arc::new(Mutex::new(Store::load()?));

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
    store: Arc<Mutex<Store>>,
}
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
        aci.create_interaction_response(http, |response| {
            response
                .kind(InteractionResponseType::ChannelMessageWithSource)
                .interaction_response_data(|message| message.content("Generating..."))
        })
        .await?;

        let prompt = util::get_value(&aci, "prompt")
            .and_then(util::value_to_string)
            .context("expected prompt")?;
        let negative_prompt =
            util::get_value(&aci, "negative_prompt").and_then(util::value_to_string);

        let models: Vec<_> = util::get_values_starting_with(&aci, "model")
            .flat_map(util::value_to_string)
            .collect();
        if models.len() > 1 {
            anyhow::bail!("more than one model specified: {:?}", models);
        }

        issue_generation_task(
            &self.client,
            &self.models,
            self.store.clone(),
            http,
            &aci,
            &sd::GenerationRequest {
                prompt: prompt.as_str(),
                negative_prompt: negative_prompt.as_deref(),
                seed: util::value_to_int(util::get_value(&aci, "seed")),
                batch_size: Some(1),
                batch_count: util::value_to_int(util::get_value(&aci, "count")).map(|v| v as u32),
                width: util::value_to_int(util::get_value(&aci, "width"))
                    .map(|v| v as u32 / 64 * 64),
                height: util::value_to_int(util::get_value(&aci, "height"))
                    .map(|v| v as u32 / 64 * 64),
                cfg_scale: util::value_to_number(util::get_value(&aci, "guidance"))
                    .map(|v| v as f32),
                steps: util::value_to_int(util::get_value(&aci, "steps")).map(|v| v as u32),
                tiling: util::value_to_bool(util::get_value(&aci, "tiling")),
                restore_faces: util::value_to_bool(util::get_value(&aci, "restore_faces")),
                sampler: util::get_value(&aci, "sampler")
                    .and_then(util::value_to_string)
                    .and_then(|v| sd::Sampler::try_from(v.as_str()).ok()),
                model: models
                    .first()
                    .and_then(|v| self.models.iter().find(|m| m.title == *v)),
                ..Default::default()
            },
        )
        .await
    }

    async fn retry(
        &self,
        http: &Http,
        mci: &MessageComponentInteraction,
        id: &str,
    ) -> anyhow::Result<()> {
        self.retry_impl(http, mci, id, None, None, Some(None)).await
    }

    async fn retry_with_options(
        &self,
        http: &Http,
        mci: &MessageComponentInteraction,
        id: &str,
    ) -> anyhow::Result<()> {
        let (old_prompt, negative_prompt, seed) = self
            .store
            .lock()
            .unwrap()
            .get(id)
            .map(|g| (g.prompt.clone(), g.negative_prompt.clone(), Some(g.seed)))
            .unwrap_or_default();

        mci.create_interaction_response(http, |r| {
            r.kind(InteractionResponseType::Modal)
                .interaction_response_data(|d| {
                    d.components(|c| {
                        c.create_action_row(|r| {
                            r.create_input_text(|t| {
                                t.label("Prompt")
                                    .custom_id("prompt")
                                    .required(true)
                                    .style(InputTextStyle::Short)
                                    .value(old_prompt)
                            })
                        })
                        .create_action_row(|r| {
                            r.create_input_text(|t| {
                                t.label("Negative prompt")
                                    .custom_id("negative_prompt")
                                    .required(false)
                                    .style(InputTextStyle::Short)
                                    .value(negative_prompt.unwrap_or_default())
                            })
                        })
                        .create_action_row(|r| {
                            r.create_input_text(|t| {
                                t.label("Seed")
                                    .custom_id("seed")
                                    .required(false)
                                    .style(InputTextStyle::Short)
                                    .value(seed.unwrap_or_default())
                            })
                        })
                    })
                    .title("Retry with prompt")
                    .custom_id(format!("{id}#retry_with_options_response"))
                })
        })
        .await?;

        Ok(())
    }

    async fn retry_with_options_response(
        &self,
        http: &Http,
        msi: &ModalSubmitInteraction,
        id: &str,
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
                    return None;
                }
            })
            .collect();

        let prompt = rows.get("prompt").map(|s| s.as_str());
        let negative_prompt = rows.get("negative_prompt").map(|s| s.as_str());
        let seed = rows.get("seed").map(|s| s.parse::<i64>().ok());

        self.retry_impl(http, msi, id, prompt, negative_prompt, seed)
            .await
    }

    async fn retry_impl(
        &self,
        http: &Http,
        interaction: &dyn GenerationInteraction,
        id: &str,
        prompt: Option<&str>,
        negative_prompt: Option<&str>,
        // None: don't override; Some(None): override with a fresh seed; Some(Some(seed)): override with seed
        seed: Option<Option<i64>>,
    ) -> anyhow::Result<()> {
        interaction.create(http).await?;

        let generation = self
            .store
            .lock()
            .unwrap()
            .get(id)
            .context("id not found in store")?
            .clone();

        let mut generation_request = generation.as_generation_request(&self.models);
        if let Some(prompt) = prompt {
            generation_request.prompt = prompt;
        }
        if let Some(negative_prompt) = negative_prompt {
            generation_request.negative_prompt = Some(negative_prompt);
        }
        if let Some(seed) = seed {
            generation_request.seed = seed;
        }

        issue_generation_task(
            &self.client,
            &self.models,
            self.store.clone(),
            http,
            interaction,
            &generation_request,
        )
        .await?;

        Ok(())
    }
}

#[async_trait]
impl EventHandler for Handler {
    async fn ready(&self, ctx: Context, ready: Ready) {
        println!("{} is connected!", ready.user.name);

        Command::create_global_application_command(&ctx.http, |command| {
            command
                .name("paint")
                .description("Paints your dreams")
                .create_option(|option| {
                    option
                        .name("prompt")
                        .description("The prompt to draw")
                        .kind(CommandOptionType::String)
                        .required(true)
                })
                .create_option(|option| {
                    option
                        .name("negative_prompt")
                        .description("The prompt to avoid drawing")
                        .kind(CommandOptionType::String)
                        .required(false)
                })
                .create_option(|option| {
                    option
                        .name("seed")
                        .description("The seed to use")
                        .kind(CommandOptionType::Integer)
                        .required(false)
                })
                .create_option(|option| {
                    option
                        .name("count")
                        .description("The number of images to generate")
                        .kind(CommandOptionType::Integer)
                        .min_int_value(1)
                        .max_int_value(4)
                        .required(false)
                })
                .create_option(|option| {
                    option
                        .name("width")
                        .description("The width of the image")
                        .kind(CommandOptionType::Integer)
                        .min_int_value(64)
                        .max_int_value(1024)
                        .required(false)
                })
                .create_option(|option| {
                    option
                        .name("height")
                        .description("The height of the image")
                        .kind(CommandOptionType::Integer)
                        .min_int_value(64)
                        .max_int_value(1024)
                        .required(false)
                })
                .create_option(|option| {
                    option
                        .name("guidance_scale")
                        .description("The scale of the guidance to apply")
                        .kind(CommandOptionType::Number)
                        .min_number_value(2.5)
                        .max_number_value(20.0)
                        .required(false)
                })
                .create_option(|option| {
                    option
                        .name("steps")
                        .description("The number of denoising steps to apply")
                        .kind(CommandOptionType::Integer)
                        .min_int_value(5)
                        .max_int_value(100)
                        .required(false)
                })
                .create_option(|option| {
                    option
                        .name("tiling")
                        .description("Whether or not the image should be tiled at the edges")
                        .kind(CommandOptionType::Boolean)
                        .required(false)
                })
                .create_option(|option| {
                    option
                        .name("restore_faces")
                        .description("Whether or not the image should have its faces restored")
                        .kind(CommandOptionType::Boolean)
                        .required(false)
                })
                .create_option(|option| {
                    let opt = option
                        .name("sampler")
                        .description("The sampler to use")
                        .kind(CommandOptionType::String)
                        .required(false);

                    for value in sd::Sampler::VALUES {
                        opt.add_string_choice(value.to_string(), value.to_string());
                    }

                    opt
                });

            for (idx, chunk) in self.models.chunks(25).enumerate() {
                command.create_option(|option| {
                    let opt = option
                        .name(if idx == 0 {
                            "model".to_string()
                        } else {
                            format!("model{}", idx + 1)
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
                .name("exilent")
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
        let result = match interaction {
            Interaction::ApplicationCommand(cmd) => {
                channel_id = Some(cmd.channel_id);
                match cmd.data.name.as_str() {
                    "paint" => self.paint(&ctx.http, cmd).await,
                    "exilent" => self.exilent(&ctx.http, cmd).await,
                    _ => Ok(()),
                }
            }
            Interaction::MessageComponent(cmp) => {
                channel_id = Some(cmp.channel_id);

                let (id, int_cmd) = cmp
                    .data
                    .custom_id
                    .split_once('#')
                    .expect("invalid interaction id");

                match int_cmd {
                    "retry" => self.retry(&ctx.http, &cmp, id).await,
                    "retry_with_options" => self.retry_with_options(&ctx.http, &cmp, id).await,
                    _ => Ok(()),
                }
            }
            Interaction::ModalSubmit(msi) => {
                channel_id = Some(msi.channel_id);

                let (id, int_cmd) = msi
                    .data
                    .custom_id
                    .split_once('#')
                    .expect("invalid interaction id");

                match int_cmd {
                    "retry_with_options_response" => {
                        self.retry_with_options_response(&ctx.http, &msi, id).await
                    }

                    _ => Ok(()),
                }
            }
            _ => Ok(()),
        };
        if let Some(channel_id) = channel_id {
            let http = &ctx.http;
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
    store: Arc<Mutex<Store>>,
    http: &Http,
    interaction: &dyn GenerationInteraction,
    request: &sd::GenerationRequest<'_>,
) -> anyhow::Result<()> {
    const PROGRESS_SCALE_FACTOR: u32 = 2;

    let prompt = request.prompt;
    let negative_prompt = request.negative_prompt;

    // generate and updat eprogress
    let task = client.generate_image_from_text(&request)?;
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
                    "{:.02}% complete. ({:.02} seconds remaining)",
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
        tokio::time::sleep(Duration::from_millis(250)).await;
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
                m.content(format!("Uploading {}/{}...", idx + 1, images.len()))
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
            image_bytes: bytes.clone(),
        };
        let message = format!(
            "{} - {}",
            generation.as_message(models),
            interaction.user().mention()
        );
        let store_key = store.lock().unwrap().insert(generation)?;

        interaction
            .channel_id()
            .send_files(&http, [(bytes.as_slice(), filename.as_str())], |m| {
                m.content(message).components(|c| {
                    c.create_action_row(|r| {
                        r.create_button(|b| {
                            b.emoji("🔃".parse::<ReactionType>().unwrap())
                                .label("Retry")
                                .style(component::ButtonStyle::Secondary)
                                .custom_id(format!("{store_key}#retry"))
                        })
                        .create_button(|b| {
                            b.emoji("↪️".parse::<ReactionType>().unwrap())
                                .label("Retry with options")
                                .style(component::ButtonStyle::Secondary)
                                .custom_id(format!("{store_key}#retry_with_options"))
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

fn encode_image_for_attachment(image: image::DynamicImage) -> anyhow::Result<Vec<u8>> {
    let mut bytes: Vec<u8> = Vec::new();
    let mut cursor = Cursor::new(&mut bytes);
    image.write_to(&mut cursor, image::ImageOutputFormat::Png)?;
    Ok(bytes)
}

#[async_trait]
trait GenerationInteraction: Send + Sync {
    async fn create(&self, http: &Http) -> anyhow::Result<()>;
    async fn get_interaction_message(&self, http: &Http) -> anyhow::Result<Message>;

    fn channel_id(&self) -> ChannelId;
    fn message(&self) -> Option<&Message>;
    fn user(&self) -> &User;
}
macro_rules! implement_interaction {
    ($name:ident) => {
        #[async_trait]
        impl GenerationInteraction for $name {
            async fn create(&self, http: &Http) -> anyhow::Result<()> {
                Ok(self
                    .create_interaction_response(http, |response| {
                        response
                            .kind(InteractionResponseType::ChannelMessageWithSource)
                            .interaction_response_data(|message| message.content("Generating..."))
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
