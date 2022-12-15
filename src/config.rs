use anyhow::Context;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, path::PathBuf};

#[derive(Serialize, Deserialize, Debug)]
pub struct General {
    pub deepdanbooru_tag_whitelist: Option<PathBuf>,
    pub automatically_prepend_keyword: bool,
    pub hide_models: HashSet<String>,
}

impl Default for General {
    fn default() -> Self {
        Self {
            deepdanbooru_tag_whitelist: Some(PathBuf::from("assets/tags/danbooru_sanitized.txt")),
            automatically_prepend_keyword: true,
            hide_models: HashSet::new(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Commands {
    pub paint: String,
    pub paintover: String,
    pub paintagain: String,
    pub postprocess: String,
    pub interrogate: String,
    pub exilent: String,
    pub png_info: String,
    pub wirehead: String,
}

impl Default for Commands {
    fn default() -> Self {
        Self {
            paint: "paint".to_string(),
            paintover: "paintover".to_string(),
            paintagain: "paintagain".to_string(),
            postprocess: "postprocess".to_string(),
            interrogate: "interrogate".to_string(),
            exilent: "exilent".to_string(),
            png_info: "pnginfo".to_string(),
            wirehead: "wirehead".to_string(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Limits {
    pub count_min: usize,
    pub count_max: usize,

    pub width_min: u32,
    pub width_max: u32,

    pub height_min: u32,
    pub height_max: u32,

    pub guidance_scale_min: f64,
    pub guidance_scale_max: f64,

    pub steps_min: usize,
    pub steps_max: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            count_min: 1,
            count_max: 4,
            width_min: 64,
            width_max: 1024,
            height_min: 64,
            height_max: 1024,
            guidance_scale_min: 2.5,
            guidance_scale_max: 20.0,
            steps_min: 5,
            steps_max: 100,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Progress {
    /// the factor to scale progress images by to reduce upload size
    pub scale_factor: f32,

    /// time in milliseonds to wait between progress updates
    pub update_ms: u64,
}

impl Default for Progress {
    fn default() -> Self {
        Self {
            scale_factor: 0.5,
            update_ms: 250,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Configuration {
    pub general: General,
    pub commands: Commands,
    pub limits: Limits,
    pub progress: Progress,
}
impl Configuration {
    const FILENAME: &str = "config.toml";

    pub fn init() -> anyhow::Result<()> {
        CONFIGURATION
            .set(Self::load()?)
            .ok()
            .context("config already set")
    }

    fn load() -> anyhow::Result<Self> {
        if let Ok(file) = std::fs::read_to_string(Self::FILENAME) {
            Ok(toml::from_str(&file)?)
        } else {
            let config = Self::default();
            config.save()?;
            Ok(config)
        }
    }

    fn save(&self) -> anyhow::Result<()> {
        Ok(std::fs::write(
            Self::FILENAME,
            &toml::to_string_pretty(self)?,
        )?)
    }
}
pub static CONFIGURATION: OnceCell<Configuration> = OnceCell::new();
