use super::simulation::{AsPhenotype, FitnessStore, TextGenome};
use crate::{constant, custom_id as cid};
use serenity::{
    http::Http,
    model::prelude::{component::ButtonStyle, ChannelId},
};
use stable_diffusion_a1111_webui_client as sd;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

// stable diffusion config
const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const CFG_SCALE: f32 = 8.0;
const SAMPLER: sd::Sampler = sd::Sampler::EulerA;
const STEPS: u32 = 15;

#[allow(clippy::too_many_arguments)]
pub async fn task(
    http: Arc<Http>,
    channel_id: ChannelId,
    client: Arc<sd::Client>,
    model: Arc<sd::Model>,
    fitness_store: Arc<FitnessStore>,
    shutdown: Arc<AtomicBool>,
    tags: Vec<String>,
    result_rx: flume::Receiver<TextGenome>,
) -> anyhow::Result<()> {
    loop {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }

        if let Ok(result) = result_rx.try_recv() {
            let image = generate(&client, (*model).clone(), result.as_text(&tags)).await?;

            channel_id
                .send_files(http.as_ref(), [(image.as_slice(), "output.png")], |m| {
                    m.content(format!(
                        "**Best result so far**: `{}`",
                        result.as_text(&tags)
                    ))
                })
                .await?;
        }

        let pending_requests = std::mem::take(&mut *fitness_store.pending_requests.lock());

        for genome in pending_requests {
            let image = generate(&client, (*model).clone(), genome.as_text(&tags)).await?;

            channel_id
                .send_files(http.as_ref(), [(image.as_slice(), "output.png")], |m| {
                    m.content(format!("`{}`", genome.as_text(&tags)))
                        .components(|mc| {
                            mc.create_action_row(|r| {
                                r.create_button(|b| {
                                    b.custom_id(cid::Wirehead::Negative2.to_id(genome.clone()))
                                        .label("-2")
                                        .style(ButtonStyle::Danger)
                                })
                                .create_button(|b| {
                                    b.custom_id(cid::Wirehead::Negative1.to_id(genome.clone()))
                                        .label("-1")
                                        .style(ButtonStyle::Danger)
                                })
                                .create_button(|b| {
                                    b.custom_id(cid::Wirehead::Zero.to_id(genome.clone()))
                                        .label("0")
                                        .style(ButtonStyle::Secondary)
                                })
                                .create_button(|b| {
                                    b.custom_id(cid::Wirehead::Positive1.to_id(genome.clone()))
                                        .label("1")
                                        .style(ButtonStyle::Success)
                                })
                                .create_button(|b| {
                                    b.custom_id(cid::Wirehead::Positive2.to_id(genome.clone()))
                                        .label("2")
                                        .style(ButtonStyle::Success)
                                })
                            })
                        })
                })
                .await?;
        }

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    Ok(())
}

async fn generate(
    client: &sd::Client,
    model: sd::Model,
    prompt: String,
) -> anyhow::Result<Vec<u8>> {
    pub fn encode_image_to_png_bytes(image: &image::DynamicImage) -> anyhow::Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut bytes);
        image.write_to(&mut cursor, image::ImageOutputFormat::Png)?;
        Ok(bytes)
    }

    let result = client
        .generate_from_text(&sd::TextToImageGenerationRequest {
            base: sd::BaseGenerationRequest {
                prompt,
                negative_prompt: None,
                batch_size: Some(1),
                batch_count: Some(1),
                width: Some(WIDTH),
                height: Some(HEIGHT),
                cfg_scale: Some(CFG_SCALE),
                denoising_strength: None,
                eta: None,
                sampler: Some(SAMPLER),
                steps: Some(STEPS),
                model: Some(model),
                ..Default::default()
            },
            ..Default::default()
        })?
        .block()
        .await;

    let image = match result {
        Ok(result) => result.images[0].clone(),
        Err(err) => {
            println!("generation failed: {:?}", err);
            image::load_from_memory(constant::resource::GENERATION_FAILED_IMAGE)?
        }
    };

    encode_image_to_png_bytes(&image)
}
