use anyhow::Context;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    io::BufRead,
    path::{Path, PathBuf},
};

use crate::constant;

#[derive(Serialize, Deserialize, Debug)]
pub struct Authentication {
    pub discord_token: Option<String>,
    pub sd_url: String,
    pub sd_api_username: Option<String>,
    pub sd_api_password: Option<String>,
}
impl Default for Authentication {
    fn default() -> Self {
        Self {
            discord_token: None,
            sd_url: "http://localhost:7860".to_string(),
            sd_api_username: None,
            sd_api_password: None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct General {
    pub deepdanbooru_tag_whitelist: Option<PathBuf>,
    pub automatically_prepend_keyword: bool,
    pub hide_models: HashSet<String>,
}
impl Default for General {
    fn default() -> Self {
        Self {
            deepdanbooru_tag_whitelist: Some(constant::resource::danbooru_sanitized_path()),
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
impl Commands {
    pub fn all(&self) -> HashSet<&str> {
        HashSet::from_iter([
            self.paint.as_str(),
            self.paintover.as_str(),
            self.paintagain.as_str(),
            self.postprocess.as_str(),
            self.interrogate.as_str(),
            self.exilent.as_str(),
            self.png_info.as_str(),
            self.wirehead.as_str(),
        ])
    }
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
pub struct Emojis {
    pub retry: String,
    pub retry_with_options: String,
    pub remix: String,
    pub upscale: String,
    pub interrogate_with_clip: String,
    pub interrogate_with_deepdanbooru: String,
    pub interrogate_generate: String,
}
impl Default for Emojis {
    fn default() -> Self {
        Self {
            retry: "🔃".to_string(),
            retry_with_options: "↪️".to_string(),
            remix: "🔀".to_string(),
            upscale: "↔".to_string(),
            interrogate_with_clip: "📋".to_string(),
            interrogate_with_deepdanbooru: "🧊".to_string(),
            interrogate_generate: "🎲".to_string(),
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
    pub authentication: Authentication,
    pub general: General,
    pub commands: Commands,
    pub emojis: Emojis,
    pub limits: Limits,
    pub progress: Progress,

    #[serde(skip)]
    runtime: ConfigurationRuntime,
}
impl Configuration {
    const FILENAME: &str = "config.toml";

    pub fn init() -> anyhow::Result<()> {
        CONFIGURATION
            .set(Self::load()?)
            .ok()
            .context("config already set")
    }

    pub fn get() -> &'static Self {
        CONFIGURATION.wait()
    }

    pub fn deepdanbooru_tag_whitelist(&self) -> Option<&Tags> {
        self.runtime.deepdanbooru_tag_whitelist.as_ref()
    }

    pub fn tags(&self) -> &HashMap<String, Tags> {
        &self.runtime.tags
    }

    fn load() -> anyhow::Result<Self> {
        let mut config = if let Ok(file) = std::fs::read_to_string(Self::FILENAME) {
            toml::from_str(&file)?
        } else {
            let config = Self::default();
            config.save()?;
            config
        };

        config.runtime = ConfigurationRuntime {
            deepdanbooru_tag_whitelist: config
                .general
                .deepdanbooru_tag_whitelist
                .as_deref()
                .map(read_tags_from_file)
                .transpose()?,
            tags: std::fs::read_dir(constant::resource::tags_dir())?
                .filter_map(|r| r.ok())
                .filter(|r| r.path().extension().unwrap_or_default() == "txt")
                .map(|de| {
                    anyhow::Ok((
                        de.path()
                            .file_stem()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string(),
                        read_tags_from_file(&de.path())?,
                    ))
                })
                .collect::<Result<HashMap<_, _>, _>>()?,
        };

        Ok(config)
    }

    fn save(&self) -> anyhow::Result<()> {
        Ok(std::fs::write(
            Self::FILENAME,
            &toml::to_string_pretty(self)?,
        )?)
    }
}
static CONFIGURATION: OnceCell<Configuration> = OnceCell::new();

pub type Tags = HashSet<String>;

#[derive(Debug, Default)]
struct ConfigurationRuntime {
    pub deepdanbooru_tag_whitelist: Option<Tags>,
    pub tags: HashMap<String, Tags>,
}

fn read_tags_from_file(path: &Path) -> anyhow::Result<Tags> {
    Ok(std::io::BufReader::new(std::fs::File::open(path)?)
        .lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.starts_with("//"))
        .collect())
}
