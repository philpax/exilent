use std::{collections::HashSet, time::Duration};

use crate::{
    cid, constant,
    store::{self, Store},
    util::{self, DiscordInteraction},
};
use serenity::{
    http::Http,
    model::prelude::{component, ReactionType},
    prelude::Mentionable,
};
use stable_diffusion_a1111_webui_client as sd;

pub async fn generation_task(
    task: sd::GenerationTask,
    models: &[sd::Model],
    store: &Store,
    http: &Http,
    interaction: &dyn DiscordInteraction,
    (prompt, negative_prompt): (&str, Option<&str>),
    image_generation: Option<store::ImageGeneration>,
) -> anyhow::Result<()> {
    use constant::misc::{PROGRESS_SCALE_FACTOR, PROGRESS_UPDATE_MS};

    // generate and update progress
    loop {
        let progress = task.progress().await?;
        let image_bytes = progress
            .current_image
            .as_ref()
            .map(|i| {
                util::encode_image_to_png_bytes(i.resize(
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
                    "`{}`{}{}: {:.02}% complete. ({:.02} seconds remaining)",
                    prompt,
                    negative_prompt
                        .map(|s| format!(" - `{s}`"))
                        .unwrap_or_default(),
                    image_generation
                        .as_ref()
                        .map(|ig| format!(" for {}", ig.init_url))
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
                        .map(|s| format!(" - `{s}`"))
                        .unwrap_or_default(),
                    idx + 1,
                    images.len()
                ),
            )
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
            denoising_strength: result.info.denoising_strength,
            image_generation: image_generation.clone(),
        };
        let message = format!(
            "{} - {}",
            generation.as_message(models),
            interaction.user().mention()
        );
        let store_key = store.insert_generation(generation)?;

        use constant::emojis as E;
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

pub async fn interrogate_task(
    client: &sd::Client,
    store: &Store,
    safe_tags: &HashSet<&str>,
    interaction: &dyn DiscordInteraction,
    http: &Http,
    (image, source, interrogator): (
        image::DynamicImage,
        store::InterrogationSource,
        sd::Interrogator,
    ),
) -> anyhow::Result<()> {
    let result = client.interrogate(&image, interrogator).await?;
    let result = if matches!(interrogator, sd::Interrogator::DeepDanbooru)
        && constant::config::USE_SAFE_TAGS
    {
        result
            .split(", ")
            .filter(|s| safe_tags.contains(s))
            .collect::<Vec<_>>()
            .join(", ")
    } else {
        result
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
                    r.create_button(|b| {
                        b.emoji(
                            constant::emojis::INTERROGATE_GENERATE
                                .parse::<ReactionType>()
                                .unwrap(),
                        )
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
                                constant::emojis::INTERROGATE_WITH_DEEPDANBOORU
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
                            b.emoji(
                                constant::emojis::INTERROGATE_WITH_CLIP
                                    .parse::<ReactionType>()
                                    .unwrap(),
                            )
                            .label("Re-interrogate with CLIP")
                            .style(component::ButtonStyle::Secondary)
                            .custom_id(cid::Interrogation::ReinterrogateWithClip.to_id(store_key))
                        }),
                    }
                })
            })
        })
        .await?;

    Ok(())
}
