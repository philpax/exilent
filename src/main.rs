use std::{
    collections::{HashMap, HashSet},
    env,
    sync::Arc,
};

use anyhow::Context as AnyhowContext;
use dotenv::dotenv;
use parking_lot::Mutex;
use serenity::{
    async_trait,
    client::{Context, EventHandler},
    model::{
        application::interaction::Interaction,
        prelude::{command::Command, *},
    },
    Client,
};

mod command;
mod constant;
mod custom_id;
mod exilent;
mod store;
mod util;
mod wirehead;

use custom_id as cid;
use stable_diffusion_a1111_webui_client as sd;
use store::Store;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();

    let client = {
        let sd_url = env::var("SD_URL").context("SD_URL not specified")?;
        let sd_authentication = env::var("SD_USER").ok().zip(env::var("SD_PASS").ok());
        Arc::new(
            sd::Client::new(
                &sd_url,
                sd_authentication
                    .as_ref()
                    .map(|p| sd::Authentication::ApiAuth(&p.0, &p.1))
                    .unwrap_or(sd::Authentication::None),
            )
            .await?,
        )
    };
    let models = client.models().await?;
    let store = Store::load()?;

    // Build our client.
    let mut client = Client::builder(
        env::var("DISCORD_TOKEN").context("Expected a token in the environment")?,
        GatewayIntents::default(),
    )
    .event_handler(Handler {
        client,
        models,
        store,
        sessions: Mutex::new(HashMap::new()),
    })
    .await
    .context("Error creating client")?;

    // Finally, start a single shard, and start listening to events.
    // Shards will automatically attempt to reconnect, and will perform
    // exponential backoff until it reconnects.
    if let Err(why) = client.start().await {
        println!("Client error: {:?}", why);
    }

    Ok(())
}

struct Handler {
    client: Arc<sd::Client>,
    models: Vec<sd::Model>,
    store: Store,
    sessions: Mutex<HashMap<ChannelId, wirehead::Session>>,
}

#[async_trait]
impl EventHandler for Handler {
    async fn ready(&self, ctx: Context, ready: Ready) {
        println!("{} is connected!", ready.user.name);

        let registered_commands: HashSet<String> =
            Command::get_global_application_commands(&ctx.http)
                .await
                .unwrap()
                .iter()
                .map(|c| c.name.clone())
                .collect();

        let our_commands: HashSet<String> = constant::command::COMMANDS
            .iter()
            .map(|s| s.to_string())
            .collect();

        if registered_commands != our_commands {
            // If the commands registered with Discord don't match the commands configured
            // for this bot, reset them entirely.
            Command::set_global_application_commands(&ctx.http, |c| {
                c.set_application_commands(vec![])
            })
            .await
            .unwrap();
        }

        exilent::command::register(&ctx.http, &self.models)
            .await
            .unwrap();
        wirehead::command::register(&ctx.http, &self.models)
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
                    constant::command::PAINT => {
                        exilent::command::paint(&self.client, &self.models, &self.store, http, cmd)
                            .await
                    }
                    constant::command::PAINTOVER => {
                        exilent::command::paintover(
                            &self.client,
                            &self.models,
                            &self.store,
                            http,
                            cmd,
                        )
                        .await
                    }
                    constant::command::PAINTAGAIN => {
                        exilent::command::paintagain(&self.store, http, cmd).await
                    }
                    constant::command::POSTPROCESS => {
                        exilent::command::postprocess(&self.client, http, cmd).await
                    }
                    constant::command::INTERROGATE => {
                        exilent::command::interrogate(&self.client, &self.store, http, cmd).await
                    }
                    constant::command::EXILENT => {
                        exilent::command::exilent(
                            &self.client,
                            &self.models,
                            &self.store,
                            http,
                            cmd,
                        )
                        .await
                    }
                    constant::command::PNG_INFO => {
                        exilent::command::png_info(&self.client, http, cmd).await
                    }
                    constant::command::WIREHEAD => {
                        wirehead::command::wirehead(
                            ctx.http.clone(),
                            cmd,
                            &self.sessions,
                            self.client.clone(),
                            &self.models,
                            &self.store,
                        )
                        .await
                    }
                    _ => Ok(()),
                }
            }
            Interaction::MessageComponent(mci) => {
                use exilent::message_component as exmc;
                use wirehead::message_component as whmc;

                channel_id = Some(mci.channel_id);

                let custom_id = cid::CustomId::try_from(mci.data.custom_id.as_str())
                    .expect("invalid interaction id");

                match custom_id {
                    cid::CustomId::Generation { id, value } => match value {
                        cid::Generation::Retry => {
                            exmc::retry(&self.client, &self.models, &self.store, http, &mci, id)
                                .await
                        }
                        cid::Generation::RetryWithOptions => {
                            exmc::retry_with_options(&self.store, http, &mci, id).await
                        }
                        cid::Generation::Remix => exmc::remix(&self.store, http, &mci, id).await,
                        cid::Generation::Upscale => {
                            exmc::upscale(&self.client, &self.store, http, &mci, id).await
                        }
                        cid::Generation::InterrogateClip => {
                            exmc::interrogate(
                                &self.client,
                                &self.store,
                                http,
                                &mci,
                                id,
                                sd::Interrogator::Clip,
                            )
                            .await
                        }
                        cid::Generation::InterrogateDeepDanbooru => {
                            exmc::interrogate(
                                &self.client,
                                &self.store,
                                http,
                                &mci,
                                id,
                                sd::Interrogator::DeepDanbooru,
                            )
                            .await
                        }
                        cid::Generation::RetryWithOptionsResponse => unreachable!(),
                        cid::Generation::RemixResponse => unreachable!(),
                    },
                    cid::CustomId::Interrogation { id, value } => match value {
                        cid::Interrogation::Generate => {
                            exmc::interrogate_generate(
                                &self.client,
                                &self.models,
                                &self.store,
                                http,
                                &mci,
                                id,
                            )
                            .await
                        }
                        cid::Interrogation::ReinterrogateWithClip => {
                            exmc::interrogate_reinterrogate(
                                &self.client,
                                &self.store,
                                http,
                                &mci,
                                id,
                                sd::Interrogator::Clip,
                            )
                            .await
                        }
                        cid::Interrogation::ReinterrogateWithDeepDanbooru => {
                            exmc::interrogate_reinterrogate(
                                &self.client,
                                &self.store,
                                http,
                                &mci,
                                id,
                                sd::Interrogator::DeepDanbooru,
                            )
                            .await
                        }
                    },
                    cid::CustomId::Wirehead { genome, value } => match value.value {
                        cid::WireheadValue::ToExilent => {
                            whmc::to_exilent(
                                &self.sessions,
                                &self.client,
                                &self.models,
                                &self.store,
                                http,
                                mci,
                                genome,
                                value.seed,
                            )
                            .await
                        }
                        _ => whmc::rate(&self.sessions, http, mci, genome, value).await,
                    },
                }
            }
            Interaction::ModalSubmit(msi) => {
                use exilent::message_component as exmc;

                channel_id = Some(msi.channel_id);

                let custom_id = cid::CustomId::try_from(msi.data.custom_id.as_str())
                    .expect("invalid interaction id");

                match custom_id {
                    cid::CustomId::Generation { id, value } => match value {
                        cid::Generation::RetryWithOptionsResponse => {
                            exmc::retry_with_options_response(
                                &self.client,
                                &self.models,
                                &self.store,
                                http,
                                &msi,
                                id,
                            )
                            .await
                        }
                        cid::Generation::RemixResponse => {
                            exmc::remix_response(
                                &self.client,
                                &self.models,
                                &self.store,
                                http,
                                &msi,
                                id,
                            )
                            .await
                        }

                        cid::Generation::Retry => unreachable!(),
                        cid::Generation::RetryWithOptions => unreachable!(),
                        cid::Generation::Remix => unreachable!(),
                        cid::Generation::Upscale => unreachable!(),
                        cid::Generation::InterrogateClip => unreachable!(),
                        cid::Generation::InterrogateDeepDanbooru => unreachable!(),
                    },
                    cid::CustomId::Interrogation { .. } => unreachable!(),
                    cid::CustomId::Wirehead { .. } => unreachable!(),
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
