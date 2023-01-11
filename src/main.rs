use anyhow::Context as AnyhowContext;
use futures::FutureExt;
use parking_lot::Mutex;
use serenity::{
    async_trait,
    client::{Context, EventHandler},
    http::Http,
    model::{
        application::interaction::Interaction,
        prelude::{command::Command, *},
    },
    Client,
};
use stable_diffusion_a1111_webui_client as sd;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

mod command;
mod config;
mod constant;
mod custom_id;
mod exilent;
mod store;
mod util;
mod wirehead;

use config::Configuration;
use custom_id as cid;
use store::Store;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    constant::resource::write_assets()?;
    Configuration::init()?;

    let authentication = &Configuration::get().authentication;
    let client = {
        let sd_authentication = Option::zip(
            authentication.sd_api_username.as_deref(),
            authentication.sd_api_password.as_deref(),
        );
        Arc::new(
            sd::Client::new(
                &authentication.sd_url,
                sd_authentication
                    .as_ref()
                    .map(|p| sd::Authentication::ApiAuth(p.0, p.1))
                    .unwrap_or(sd::Authentication::None),
            )
            .await?,
        )
    };
    let models: Vec<_> = client
        .models()
        .await?
        .into_iter()
        .filter(|m| {
            !Configuration::get()
                .general
                .hide_models
                .contains(util::extract_last_bracketed_string(&m.title).unwrap())
        })
        .collect();
    let store = Store::load()?;

    // Build our client.
    let mut client = Client::builder(
        authentication
            .discord_token
            .as_deref()
            .context("Expected authentication.discord_token to be filled in config")?,
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

async fn ready_handler(http: &Http, models: &[sd::Model]) -> anyhow::Result<()> {
    let registered_commands = Command::get_global_application_commands(http).await?;
    let registered_commands: HashSet<_> = registered_commands
        .iter()
        .map(|c| c.name.as_str())
        .collect();

    let our_commands: HashSet<_> = Configuration::get()
        .commands
        .all()
        .iter()
        .cloned()
        .collect();

    if registered_commands != our_commands {
        // If the commands registered with Discord don't match the commands configured
        // for this bot, reset them entirely.
        Command::set_global_application_commands(http, |c| c.set_application_commands(vec![]))
            .await?;
    }

    // TEMP HACK: Serenity 0.11.5 does not handle top-level error objects correctly,
    // and will panic if it can't parse an error body. We catch the panic and lift it
    // to an error.
    //
    // https://github.com/serenity-rs/serenity/pull/2256 fixes this.
    async fn catch_async_panic(
        f: impl std::future::Future<Output = anyhow::Result<()>>,
    ) -> anyhow::Result<()> {
        use std::panic;

        let prev_hook = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let r = panic::AssertUnwindSafe(f)
            .catch_unwind()
            .await
            .map_err(|e| {
                anyhow::anyhow!(
                    "{}",
                    match e.downcast_ref::<String>() {
                        Some(v) => v.as_str(),
                        None => match e.downcast_ref::<&str>() {
                            Some(v) => v,
                            _ => "Unknown Source of Error",
                        },
                    }
                )
            })?;
        panic::set_hook(prev_hook);
        r
    }
    catch_async_panic(async {
        exilent::command::register(http, models).await?;
        wirehead::command::register(http, models).await?;

        anyhow::Ok(())
    })
    .await?;

    Ok(())
}

#[async_trait]
impl EventHandler for Handler {
    async fn ready(&self, ctx: Context, ready: Ready) {
        println!("{} is connected; registering commands...", ready.user.name);

        if let Err(err) = ready_handler(&ctx.http, &self.models).await {
            println!("Error while registering commands: `{}`", err);
            if err.to_string() == "expected object" {
                println!();
                println!(
                    "Discord refused to register the commands due to the request being too long."
                );
                println!(
                    "Consider hiding some models under `general.hide_models` in `config.toml` to fix this."
                );
                ctx.shard.shutdown_clean();
                std::process::exit(1);
            }
        }

        println!("{} is good to go!", ready.user.name);
    }

    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        let http = &ctx.http;
        match interaction {
            Interaction::ApplicationCommand(cmd) => {
                let name = cmd.data.name.as_str();
                let commands = &Configuration::get().commands;

                if name == commands.paint {
                    exilent::command::paint(&self.client, &self.models, &self.store, http, cmd)
                        .await
                } else if name == commands.paintover {
                    exilent::command::paintover(&self.client, &self.models, &self.store, http, cmd)
                        .await
                } else if name == commands.postprocess {
                    exilent::command::postprocess(&self.client, http, cmd).await
                } else if name == commands.interrogate {
                    exilent::command::interrogate(&self.client, &self.store, http, cmd).await
                } else if name == commands.exilent {
                    exilent::command::exilent(&self.client, &self.models, &self.store, http, cmd)
                        .await
                } else if name == commands.png_info {
                    exilent::command::png_info(&self.client, http, cmd).await
                } else if name == commands.wirehead {
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
            }
            Interaction::MessageComponent(mci) => {
                use exilent::message_component as exmc;
                use wirehead::message_component as whmc;

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
                                &self.store,
                                (&self.client, &self.models),
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
            _ => {}
        };
    }
}
