use super::simulation::{AsPhenotype, FitnessStore, TextGenome};
use crate::{command::OwnedBaseGenerationParameters, constant, custom_id as cid, util};
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

#[allow(clippy::too_many_arguments)]
pub async fn task(
    http: Arc<Http>,
    channel_id: ChannelId,
    to_exilent_enabled: bool,
    client: Arc<sd::Client>,
    params: OwnedBaseGenerationParameters,
    fitness_store: Arc<FitnessStore>,
    shutdown: Arc<AtomicBool>,
    tags: Vec<String>,
    prefix: Option<String>,
    suffix: Option<String>,
    result_rx: flume::Receiver<TextGenome>,
    hide_prompt: bool,
) -> anyhow::Result<()> {
    loop {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }

        fn to_attachment_type(value: &(Vec<u8>, i64)) -> AttachmentType {
            AttachmentType::Bytes {
                data: Cow::Borrowed(value.0.as_slice()),
                filename: format!("output_{}.png", value.1),
            }
        }

        if let Ok(genome) = result_rx.try_recv() {
            let prompt = genome.as_text(&tags, prefix.as_deref(), suffix.as_deref());
            let images =
                generate(&client, params.as_base_generation_request(), prompt.clone()).await?;

            channel_id
                .send_files(http.as_ref(), images.iter().map(to_attachment_type), |m| {
                    m.content(format!(
                        "**Best result so far**{}",
                        if !hide_prompt {
                            format!(": `{}`", prompt)
                        } else {
                            String::new()
                        }
                    ))
                    .components(|c| {
                        if to_exilent_enabled {
                            c.create_action_row(|row| {
                                row.create_button(|b| {
                                    b.custom_id(
                                        cid::WireheadValue::ToExilent.to_id(genome, images[0].1),
                                    )
                                    .label("To Exilent")
                                    .style(ButtonStyle::Primary)
                                })
                            })
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
                params.as_base_generation_request(),
                genome.as_text(&tags, prefix.as_deref(), suffix.as_deref()),
            )
            .await?;

            let seed = images[0].1;

            channel_id
                .send_files(http.as_ref(), images.iter().map(to_attachment_type), |m| {
                    m.components(|mc| {
                        mc.create_action_row(|row| {
                            let g = &genome;
                            row.create_button(|b| {
                                b.custom_id(cid::WireheadValue::Negative2.to_id(g.clone(), seed))
                                    .label("-2")
                                    .style(ButtonStyle::Danger)
                            })
                            .create_button(|b| {
                                b.custom_id(cid::WireheadValue::Negative1.to_id(g.clone(), seed))
                                    .label("-1")
                                    .style(ButtonStyle::Danger)
                            })
                            .create_button(|b| {
                                b.custom_id(cid::WireheadValue::Zero.to_id(g.clone(), seed))
                                    .label("0")
                                    .style(ButtonStyle::Secondary)
                            })
                            .create_button(|b| {
                                b.custom_id(cid::WireheadValue::Positive1.to_id(g.clone(), seed))
                                    .label("1")
                                    .style(ButtonStyle::Success)
                            })
                            .create_button(|b| {
                                b.custom_id(cid::WireheadValue::Positive2.to_id(g.clone(), seed))
                                    .label("2")
                                    .style(ButtonStyle::Success)
                            })
                        })
                    });

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
    base_generation_request: sd::BaseGenerationRequest,
    prompt: String,
) -> anyhow::Result<Vec<(Vec<u8>, i64)>> {
    let mut base = sd::BaseGenerationRequest {
        prompt,
        ..base_generation_request
    };
    util::fixup_base_generation_request(&mut base);

    let result = client
        .generate_from_text(&sd::TextToImageGenerationRequest {
            base,
            ..Default::default()
        })
        .await;

    Ok(match result {
        Ok(result) => result.pngs.into_iter().zip(result.info.seeds).collect(),
        Err(err) => {
            println!("generation failed: {:?}", err);
            vec![(
                util::encode_image_to_png_bytes(image::open(
                    constant::resource::generation_failed_path(),
                )?)?,
                0,
            )]
        }
    })
}
