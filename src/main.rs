use std::{env, io::Cursor};

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
            interaction::application_command::{
                ApplicationCommandInteraction, CommandDataOptionValue,
            },
            *,
        },
    },
    Client,
};

use stable_diffusion_a1111_webui_client as sd;

struct Handler(sd::Client);

async fn handle_generation(
    client: &sd::Client,
    command: &ApplicationCommandInteraction,
    channel: ChannelId,
    http: &Http,
) -> anyhow::Result<()> {
    let options = &command.data.options;

    let get_value = |name: &str| {
        options
            .iter()
            .find(|v| v.name == name)
            .and_then(|v| v.resolved.as_ref())
    };
    let value_to_int = |v: Option<&CommandDataOptionValue>| match v? {
        CommandDataOptionValue::Integer(v) => Some(*v),
        _ => None,
    };
    let value_to_number = |v: Option<&CommandDataOptionValue>| match v? {
        CommandDataOptionValue::Number(v) => Some(*v),
        _ => None,
    };
    let value_to_string = |v: Option<&CommandDataOptionValue>| match v? {
        CommandDataOptionValue::String(v) => Some(v.clone()),
        _ => None,
    };
    let value_to_bool = |v: Option<&CommandDataOptionValue>| match v? {
        CommandDataOptionValue::Boolean(v) => Some(*v),
        _ => None,
    };

    let prompt = value_to_string(get_value("prompt")).context("expected prompt")?;
    let seed = value_to_int(get_value("seed"));
    let count = value_to_int(get_value("count")).map(|v| v as u32);
    let width = value_to_int(get_value("width")).map(|v| v as u32 / 64 * 64);
    let height = value_to_int(get_value("height")).map(|v| v as u32 / 64 * 64);
    let guidance = value_to_number(get_value("guidance")).map(|v| v as f32);
    let steps = value_to_int(get_value("steps")).map(|v| v as u32);
    let tiling = value_to_bool(get_value("tiling"));
    let restore_faces = value_to_bool(get_value("restore_faces"));
    let sampler =
        value_to_string(get_value("sampler")).and_then(|v| sd::Sampler::try_from(v.as_str()).ok());

    let result = client
        .generate_image_from_text(&sd::GenerationRequest {
            prompt: prompt.as_str(),
            seed,
            batch_size: Some(1),
            batch_count: count,
            width,
            height,
            cfg_scale: guidance,
            steps,
            tiling,
            restore_faces,
            sampler,
            ..Default::default()
        })
        .block()
        .await?;

    let images = result
        .images
        .into_iter()
        .enumerate()
        .map(|(idx, image)| {
            let mut bytes: Vec<u8> = Vec::new();
            let mut cursor = Cursor::new(&mut bytes);
            image.write_to(&mut cursor, image::ImageOutputFormat::Png)?;
            Ok((format!("image_{idx}.png"), bytes))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    channel
        .send_files(
            &http,
            images
                .iter()
                .map(|(name, bytes)| (bytes.as_slice(), name.as_str()))
                .collect::<Vec<_>>(),
            |m| {
                m.content(
                    result
                        .info
                        .seeds
                        .iter()
                        .map(|seed| format!("`/paint prompt:{prompt} seed:{seed} count:{} width:{} height:{} guidance_scale:{} steps:{} tiling:{} restore_faces:{} sampler:{}`", result.info.seeds.len(), result.info.width, result.info.height, result.info.cfg_scale, result.info.steps, result.info.tiling, result.info.restore_faces, result.info.sampler.to_string()))
                        .collect::<Vec<_>>()
                        .join("\n"),
                )
            },
        )
        .await?;

    Ok(())
}

#[async_trait]
impl EventHandler for Handler {
    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        let Interaction::ApplicationCommand(command) = interaction else { return };

        if command.data.name != "paint" {
            return;
        }
        let channel = command.channel_id;

        let result = command
            .create_interaction_response(&ctx.http, |response| {
                response
                    .kind(InteractionResponseType::ChannelMessageWithSource)
                    .interaction_response_data(|message| message.content("Generating..."))
            })
            .await;
        if let Err(why) = result {
            println!("Cannot respond to slash command (pre-generation): {}", why);
            return;
        }

        if let Err(err) = handle_generation(&self.0, &command, channel, &ctx.http).await {
            channel
                .send_message(&ctx.http, |r| r.content(format!("Error: {err:?}")))
                .await
                .ok();
        }
    }

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
                })
        })
        .await
        .unwrap();
    }
}

#[tokio::main]
async fn main() {
    dotenv().ok();
    // Configure the client with your Discord bot token in the environment.
    let token = env::var("DISCORD_TOKEN").expect("Expected a token in the environment");

    let sd_url = env::var("SD_URL").expect("SD_URL not specified");
    let sd_authentication = env::var("SD_USER").ok().zip(env::var("SD_PASS").ok());

    let client = sd::Client::new(&sd_url, sd_authentication.as_ref().map(|p| (&*p.0, &*p.1)))
        .await
        .unwrap();

    // Build our client.
    let intents = GatewayIntents::default();

    let mut client = Client::builder(token, intents)
        .event_handler(Handler(client))
        .await
        .expect("Error creating client");

    // Finally, start a single shard, and start listening to events.
    // Shards will automatically attempt to reconnect, and will perform
    // exponential backoff until it reconnects.
    if let Err(why) = client.start().await {
        println!("Client error: {:?}", why);
    }
}
