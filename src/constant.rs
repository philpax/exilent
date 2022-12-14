/// names of values used in interactions
pub mod value {
    pub const PROMPT: &str = "prompt";
    pub const NEGATIVE_PROMPT: &str = "negative_prompt";
    pub const SEED: &str = "seed";
    pub const COUNT: &str = "count";
    pub const WIDTH: &str = "width";
    pub const HEIGHT: &str = "height";
    pub const GUIDANCE_SCALE: &str = "guidance_scale";
    pub const STEPS: &str = "steps";
    pub const TILING: &str = "tiling";
    pub const RESTORE_FACES: &str = "restore_faces";
    pub const SAMPLER: &str = "sampler";
    pub const MODEL: &str = "model";
    pub const DENOISING_STRENGTH: &str = "denoising_strength";

    pub const WIDTH_HEIGHT: &str = "width_height";
    pub const GUIDANCE_SCALE_DENOISING_STRENGTH: &str = "guidance_scale_denoising_strength";

    pub const RESIZE_MODE: &str = "resize_mode";

    pub const UPSCALER_1: &str = "upscaler_1";
    pub const UPSCALER_2: &str = "upscaler_2";
    pub const SCALE_FACTOR: &str = "scale_factor";
    pub const CODEFORMER_VISIBILITY: &str = "codeformer_visibility";
    pub const CODEFORMER_WEIGHT: &str = "codeformer_weight";
    pub const UPSCALER_2_VISIBILITY: &str = "upscaler_2_visibility";
    pub const GFPGAN_VISIBILITY: &str = "gfpgan_visibility";
    pub const UPSCALE_FIRST: &str = "upscale_first";

    pub const IMAGE_URL: &str = "image_url";
    pub const IMAGE_ATTACHMENT: &str = "image_attachment";
    pub const INTERROGATOR: &str = "interrogator";

    pub const TAGS: &str = "tags";
    pub const HIDE_PROMPT: &str = "hide_prompt";
    pub const TO_EXILENT_ENABLED: &str = "to_exilent_enabled";
    pub const TO_EXILENT_CHANNEL: &str = "to_exilent_channel";
    pub const PREFIX: &str = "prefix";
    pub const SUFFIX: &str = "suffix";

    /// Discord allows for a maximum of 25 options in a choice
    pub const MODEL_CHUNK_COUNT: usize = 25;
}

/// resource
pub mod resource {
    use std::{collections::HashMap, path::PathBuf};

    pub fn assets_dir() -> PathBuf {
        PathBuf::from("assets")
    }

    pub fn generation_failed_path() -> PathBuf {
        assets_dir().join("generation_failed.png")
    }

    pub fn tags_dir() -> PathBuf {
        assets_dir().join("tags")
    }

    pub fn danbooru_sanitized_path() -> PathBuf {
        tags_dir().join("danbooru_sanitized.txt")
    }

    pub fn write_assets() -> anyhow::Result<()> {
        fn write_file(path: PathBuf, contents: Vec<u8>) -> anyhow::Result<()> {
            if !path.exists() {
                std::fs::write(path, contents)?;
            }
            Ok(())
        }

        let resources: HashMap<PathBuf, String> =
            serde_json::from_str(include_str!(concat!(env!("OUT_DIR"), "/resource.json")))?;

        let assets_dir = assets_dir();
        for (path, contents) in resources {
            let path = assets_dir.join(path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            write_file(path, hex::decode(contents)?)?;
        }

        Ok(())
    }
}
