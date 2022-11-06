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
    let CommandDataOptionValue::String(prompt) = options
        .get(0)
        .and_then(|v| v.resolved.as_ref())
        .context("expected prompt")? else { anyhow::bail!("invalid prompt type"); };
    let seed = options
        .get(1)
        .and_then(|v| v.resolved.as_ref())
        .and_then(|v| match v {
            CommandDataOptionValue::Integer(seed) => Some(*seed),
            _ => None,
        });

    let result = client
        .generate_image_from_text(&sd::GenerationRequest {
            prompt: prompt.as_str(),
            seed,
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
                        .into_iter()
                        .map(|seed| format!("`/paint prompt:{prompt} seed:{seed}`"))
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
