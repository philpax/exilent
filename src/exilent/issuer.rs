use crate::{
    cid,
    config::Configuration,
    store::{self, Store},
    util::{self, DiscordInteraction},
};
use anyhow::Context;
use serenity::{
    http::Http,
    model::prelude::{component, ChannelId, ReactionType},
    prelude::Mentionable,
};
use stable_diffusion_a1111_webui_client as sd;
use std::time::Duration;

#[allow(clippy::too_many_arguments)]
pub async fn generation_task(
    client: &sd::Client,
    task: tokio::task::JoinHandle<sd::Result<sd::GenerationResult>>,
    models: &[sd::Model],
    store: &Store,
    http: &Http,
    interaction: &dyn DiscordInteraction,
    result_channel_override: Option<ChannelId>,
    (prompt, negative_prompt): (&str, Option<&str>),
    image_generation: Option<store::ImageGeneration>,
) -> anyhow::Result<()> {
    // How many seconds to subtract from the time of job issuance to accommodate for
    // early starts
    const START_TIME_SLACK: i64 = 2;

    // generate and update progress
    let mut max_progress_factor = 0.0;

    let start_time = chrono::Local::now() - chrono::Duration::seconds(START_TIME_SLACK);

    loop {
        let progress = client.progress().await?;

        // Only update the message if the ongoing job was started after
        // this job was issued
        if progress.job_timestamp.unwrap_or(start_time) >= start_time {
            let image_bytes = progress
                .current_image
                .as_ref()
                .map(|i| {
                    util::encode_image_to_png_bytes(i.resize(
                        ((i.width() as f32) * Configuration::get().progress.scale_factor) as u32,
                        ((i.height() as f32) * Configuration::get().progress.scale_factor) as u32,
                        image::imageops::FilterType::Nearest,
                    ))
                })
                .transpose()?;

            max_progress_factor = progress.progress_factor.max(max_progress_factor);

            interaction
                .get_interaction_message(http)
                .await?
                .edit(http, |m| {
                    m.content(format!(
                        "`{}`{}{}: {:.02}% complete. ({:.02} seconds remaining)",
                        prompt,
                        negative_prompt
                            .filter(|s| !s.is_empty())
                            .map(|s| format!(" - `{s}`"))
                            .unwrap_or_default(),
                        image_generation
                            .as_ref()
                            .map(|ig| format!(" for {}", ig.init_url))
                            .unwrap_or_default(),
                        max_progress_factor * 100.0,
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
        }

        if task.is_finished() {
            break;
        }

        tokio::time::sleep(Duration::from_millis(
            Configuration::get().progress.update_ms,
        ))
        .await;
    }

    // retrieve result
    let result = task.await??;
    let images = result
        .images
        .into_iter()
        .enumerate()
        .map(|(idx, image)| {
            Ok((
                format!("image_{idx}.png"),
                util::encode_image_to_png_bytes(image)?,
            ))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // send images
    for (idx, ((filename, bytes), seed)) in images.iter().zip(result.info.seeds.iter()).enumerate()
    {
        interaction
            .edit(
                http,
                &format!(
                    "`{}`{}: Uploading {}/{}...",
                    prompt,
                    negative_prompt
                        .filter(|s| !s.is_empty())
                        .map(|s| format!(" - `{s}`"))
                        .unwrap_or_default(),
                    idx + 1,
                    images.len()
                ),
            )
            .await?;

        let generation = store::Generation {
            id: None,
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
            image_url: None,
            timestamp: result.info.job_timestamp,
            user_id: interaction.user().id,
            denoising_strength: result.info.denoising_strength,
            image_generation: image_generation.clone(),
        };
        let message = format!(
            "{} - {}",
            generation.as_message(models),
            interaction.user().mention()
        );
        let store_key = store.insert_generation(generation)?;

        let final_message = result_channel_override
            .unwrap_or_else(|| interaction.channel_id())
            .send_files(&http, [(bytes.as_slice(), filename.as_str())], |m| {
                m.content(message).components(|c| {
                    let e = &Configuration::get().emojis;
                    c.create_action_row(|r| {
                        r.create_button(|b| {
                            b.emoji(e.retry.parse::<ReactionType>().unwrap())
                                .label("Retry")
                                .style(component::ButtonStyle::Secondary)
                                .custom_id(cid::Generation::Retry.to_id(store_key))
                        })
                        .create_button(|b| {
                            b.emoji(e.retry_with_options.parse::<ReactionType>().unwrap())
                                .label("Retry (options)")
                                .style(component::ButtonStyle::Secondary)
                                .custom_id(cid::Generation::RetryWithOptions.to_id(store_key))
                        })
                        .create_button(|b| {
                            b.emoji(e.remix.parse::<ReactionType>().unwrap())
                                .label("Remix")
                                .style(component::ButtonStyle::Secondary)
                                .custom_id(cid::Generation::Remix.to_id(store_key))
                        })
                        .create_button(|b| {
                            b.emoji(e.upscale.parse::<ReactionType>().unwrap())
                                .label("Upscale (ESRGAN 2x)")
                                .style(component::ButtonStyle::Secondary)
                                .custom_id(cid::Generation::Upscale.to_id(store_key))
                        })
                    })
                    .create_action_row(|r| {
                        r.create_button(|b| {
                            b.emoji(e.interrogate_with_clip.parse::<ReactionType>().unwrap())
                                .label("CLIP")
                                .style(component::ButtonStyle::Secondary)
                                .custom_id(cid::Generation::InterrogateClip.to_id(store_key))
                        })
                        .create_button(|b| {
                            b.emoji(
                                e.interrogate_with_deepdanbooru
                                    .parse::<ReactionType>()
                                    .unwrap(),
                            )
                            .label("DeepDanbooru")
                            .style(component::ButtonStyle::Secondary)
                            .custom_id(cid::Generation::InterrogateDeepDanbooru.to_id(store_key))
                        })
                    })
                });

                if result_channel_override.is_none() {
                    if let Some(message) = interaction.message() {
                        m.reference_message(message);
                    }
                }

                m
            })
            .await?;

        store.set_generation_url(
            store_key,
            &final_message
                .attachments
                .first()
                .context("no attachment")?
                .url,
        )?;
    }
    interaction
        .get_interaction_message(http)
        .await?
        .delete(http)
        .await?;

    Ok(())
}

pub async fn interrogate_task(
    client: &sd::Client,
    store: &Store,
    interaction: &dyn DiscordInteraction,
    http: &Http,
    (image, source, interrogator): (
        image::DynamicImage,
        store::InterrogationSource,
        sd::Interrogator,
    ),
) -> anyhow::Result<()> {
    let result = client.interrogate(&image, interrogator).await?;
    let result = match (
        interrogator,
        Configuration::get().deepdanbooru_tag_whitelist(),
    ) {
        (sd::Interrogator::DeepDanbooru, Some(tags)) => result
            .split(", ")
            .filter(|s| tags.contains(*s))
            .collect::<Vec<_>>()
            .join(", "),
        _ => result,
    };

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
                    let e = &Configuration::get().emojis;
                    r.create_button(|b| {
                        b.emoji(e.interrogate_generate.parse::<ReactionType>().unwrap())
                            .label(match interrogator {
                                sd::Interrogator::Clip => "Generate",
                                sd::Interrogator::DeepDanbooru => "Generate with shuffle",
                            })
                            .style(component::ButtonStyle::Secondary)
                            .custom_id(cid::Interrogation::Generate.to_id(store_key))
                    });

                    match interrogator {
                        sd::Interrogator::Clip => r.create_button(|b| {
                            b.emoji(
                                e.interrogate_with_deepdanbooru
                                    .parse::<ReactionType>()
                                    .unwrap(),
                            )
                            .label("Re-interrogate with DeepDanbooru")
                            .style(component::ButtonStyle::Secondary)
                            .custom_id(
                                cid::Interrogation::ReinterrogateWithDeepDanbooru.to_id(store_key),
                            )
                        }),
                        sd::Interrogator::DeepDanbooru => r.create_button(|b| {
                            b.emoji(e.interrogate_with_clip.parse::<ReactionType>().unwrap())
                                .label("Re-interrogate with CLIP")
                                .style(component::ButtonStyle::Secondary)
                                .custom_id(
                                    cid::Interrogation::ReinterrogateWithClip.to_id(store_key),
                                )
                        }),
                    }
                })
            })
        })
        .await?;

    Ok(())
}
