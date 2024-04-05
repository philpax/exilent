use super::{
    simulation::{AsPhenotype, FitnessStore, TextGenome},
    GenerationParameters,
};
use crate::{
    command::GenerationParameters as CommandGenerationParameters, constant, custom_id as cid, util,
};
use serenity::{
    http::Http,
    model::prelude::{component::ButtonStyle, AttachmentType, ChannelId},
};
use stable_diffusion_a1111_webui_client as sd;
use std::{
    borrow::Cow,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

pub struct Parameters {
    pub http: Arc<Http>,
    pub channel_id: ChannelId,

    pub shutdown: Arc<AtomicBool>,

    pub fitness_store: Arc<FitnessStore>,
    pub result_rx: flume::Receiver<TextGenome>,

    pub to_exilent_enabled: bool,
    pub hide_prompt: bool,

    pub client: Arc<sd::Client>,
    pub generation_parameters: GenerationParameters,
}

pub async fn task(parameters: Parameters) -> anyhow::Result<()> {
    let Parameters {
        http,
        channel_id,
        shutdown,
        fitness_store,
        result_rx,
        to_exilent_enabled,
        hide_prompt,
        client,
        generation_parameters,
    } = parameters;

    let GenerationParameters {
        parameters,
        tags,
        prefix,
        suffix,
    } = generation_parameters;

    loop {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }

        fn to_attachment_type(value: &(Vec<u8>, Option<i64>)) -> AttachmentType {
            AttachmentType::Bytes {
                data: Cow::Borrowed(value.0.as_slice()),
                filename: format!("output_{}.png", value.1.unwrap_or_default()),
            }
        }

        if let Ok(genome) = result_rx.try_recv() {
            let prompt = genome.as_text(&tags, prefix.as_deref(), suffix.as_deref());
            let images = generate(&client, parameters.clone(), prompt.clone()).await?;

            channel_id
                .send_files(http.as_ref(), images.iter().map(to_attachment_type), |m| {
                    m.content(format!(
                        "**Best result so far**{}",
                        if !hide_prompt {
                            format!(": `{prompt}`")
                        } else {
                            String::new()
                        }
                    ))
                    .components(|c| {
                        if to_exilent_enabled {
                            match images.first().and_then(|i| i.1) {
                                Some(seed) => c.create_action_row(|row| {
                                    row.create_button(|b| {
                                        b.custom_id(
                                            cid::WireheadValue::ToExilent.to_id(genome, seed),
                                        )
                                        .label("To Exilent")
                                        .style(ButtonStyle::Primary)
                                    })
                                }),
                                None => c,
                            }
                        } else {
                            c
                        }
                    })
                })
                .await?;
        }

        let pending_requests = std::mem::take(&mut *fitness_store.pending_requests.lock());

        for genome in pending_requests {
            let images = generate(
                &client,
                parameters.clone(),
                genome.as_text(&tags, prefix.as_deref(), suffix.as_deref()),
            )
            .await?;

            channel_id
                .send_files(http.as_ref(), images.iter().map(to_attachment_type), |m| {
                    if let Some(seed) = images.first().and_then(|i| i.1) {
                        m.components(|mc| {
                            mc.create_action_row(|row| {
                                let g = &genome;
                                row.create_button(|b| {
                                    b.custom_id(
                                        cid::WireheadValue::Negative2.to_id(g.clone(), seed),
                                    )
                                    .label("-2")
                                    .style(ButtonStyle::Danger)
                                })
                                .create_button(|b| {
                                    b.custom_id(
                                        cid::WireheadValue::Negative1.to_id(g.clone(), seed),
                                    )
                                    .label("-1")
                                    .style(ButtonStyle::Danger)
                                })
                                .create_button(|b| {
                                    b.custom_id(cid::WireheadValue::Zero.to_id(g.clone(), seed))
                                        .label("0")
                                        .style(ButtonStyle::Secondary)
                                })
                                .create_button(|b| {
                                    b.custom_id(
                                        cid::WireheadValue::Positive1.to_id(g.clone(), seed),
                                    )
                                    .label("1")
                                    .style(ButtonStyle::Success)
                                })
                                .create_button(|b| {
                                    b.custom_id(
                                        cid::WireheadValue::Positive2.to_id(g.clone(), seed),
                                    )
                                    .label("2")
                                    .style(ButtonStyle::Success)
                                })
                            })
                        });
                    }

                    if !hide_prompt {
                        m.content(format!(
                            "`{}`",
                            genome.as_text(&tags, prefix.as_deref(), suffix.as_deref())
                        ));
                    }

                    m
                })
                .await?;
        }

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    Ok(())
}

/// always guaranteed to return at least one image if it suceeds
async fn generate(
    client: &sd::Client,
    mut parameters: CommandGenerationParameters,
    prompt: String,
) -> anyhow::Result<Vec<(Vec<u8>, Option<i64>)>> {
    parameters.base_generation_mut().prompt = prompt;
    let result = parameters.generate(client).await;

    Ok(match result {
        Ok(result) => result
            .pngs
            .into_iter()
            .zip(result.info.seeds.into_iter().map(Some))
            .collect(),
        Err(err) => {
            println!("generation failed: {err:?}");
            vec![(
                util::encode_image_to_png_bytes(image::open(
                    constant::resource::generation_failed_path(),
                )?)?,
                None,
            )]
        }
    })
}
