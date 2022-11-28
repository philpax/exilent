use std::collections::HashMap;

use crate::{sd, util};
use serde::{Deserialize, Serialize};
use stable_diffusion_a1111_webui_client::Sampler;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Generation {
    pub prompt: String,
    pub seed: i64,
    pub width: u32,
    pub height: u32,
    pub cfg_scale: f32,
    pub steps: u32,
    pub tiling: bool,
    pub restore_faces: bool,
    pub sampler: Sampler,
    pub negative_prompt: Option<String>,
    pub model_hash: String,
    pub image_bytes: Vec<u8>,
}
impl Generation {
    pub fn as_message(&self, models: &[sd::Model]) -> String {
        format!("`/paint prompt:{} seed:{} count:1 width:{} height:{} guidance_scale:{} steps:{} tiling:{} restore_faces:{} sampler:{} {} {}`",
            self.prompt,
            self.seed,
            self.width,
            self.height,
            self.cfg_scale,
            self.steps,
            self.tiling,
            self.restore_faces,
            self.sampler.to_string(),
            self.negative_prompt.as_ref().map(|s| format!("negative_prompt:{s}")).unwrap_or_default(),
            util::find_model_by_hash(models, &self.model_hash).map(|m| format!("model:{}", m.name)).unwrap_or_default()
        )
    }

    pub fn as_generation_request<'a>(
        &'a self,
        models: &'a [sd::Model],
    ) -> sd::GenerationRequest<'a> {
        sd::GenerationRequest {
            prompt: self.prompt.as_str(),
            negative_prompt: self.negative_prompt.as_ref().map(|s| s.as_str()),
            seed: Some(self.seed),
            batch_size: Some(1),
            batch_count: Some(1),
            width: Some(self.width),
            height: Some(self.height),
            cfg_scale: Some(self.cfg_scale),
            steps: Some(self.steps),
            tiling: Some(self.tiling),
            restore_faces: Some(self.restore_faces),
            sampler: Some(self.sampler),
            model: util::find_model_by_hash(models, &self.model_hash),
            ..Default::default()
        }
    }
}

type StoreValue = Generation;
pub struct Store(HashMap<String, StoreValue>);
impl Store {
    const FILENAME: &str = "store.json";

    pub fn load() -> anyhow::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .read(true)
            .create(true)
            .open(Self::FILENAME)?;

        let data = std::io::read_to_string(file)?;
        Ok(Self(if data.is_empty() {
            HashMap::new()
        } else {
            serde_json::from_str(&data)?
        }))
    }

    pub fn save(&self) -> anyhow::Result<()> {
        Ok(std::fs::write(
            Self::FILENAME,
            serde_json::to_string(&self.0)?,
        )?)
    }

    pub fn insert(&mut self, value: StoreValue) -> anyhow::Result<String> {
        let key = nanoid::nanoid!();
        self.0.insert(key.clone(), value);
        self.save()?;
        Ok(key)
    }

    pub fn get(&self, key: &str) -> Option<&StoreValue> {
        self.0.get(key)
    }
}
