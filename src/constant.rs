/// config
pub mod config {
    /// If set, Exilent will remove any tags in DeepDanbooru results that aren't
    /// present in safe_tags.txt.
    pub const USE_SAFE_TAGS: bool = true;

    /// If set, Exilent will automatically look for a keyword in the square brackets
    /// of a model name and prepend it to the prompt if it is not already present.
    ///
    /// Inkpunk v2 [nvinkpunk] will add `nvinkpunk` to the start of a prompt.
    /// This does not work if there is more than one keyword.
    pub const AUTOMATICALLY_PREPEND_KEYWORD: bool = true;
}

/// discord command names
pub mod command {
    pub const PAINT: &str = "paint";
    pub const PAINTOVER: &str = "paintover";
    pub const PAINTAGAIN: &str = "paintagain";
    pub const POSTPROCESS: &str = "postprocess";
    pub const INTERROGATE: &str = "interrogate";
    pub const EXILENT: &str = "exilent";
    pub const PNG_INFO: &str = "pnginfo";
}

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
}

/// emojis
pub mod emojis {
    pub const RETRY: &str = "🔃";
    pub const RETRY_WITH_OPTIONS: &str = "↪️";
    pub const REMIX: &str = "🔀";
    pub const UPSCALE: &str = "↔";
    pub const INTERROGATE_WITH_CLIP: &str = "📋";
    pub const INTERROGATE_WITH_DEEPDANBOORU: &str = "🧊";

    pub const INTERROGATE_GENERATE: &str = "🎲";
}

/// limits
pub mod limits {
    pub const COUNT_MIN: usize = 1;
    pub const COUNT_MAX: usize = 4;

    pub const WIDTH_MIN: u32 = 64;
    pub const WIDTH_MAX: u32 = 1024;

    pub const HEIGHT_MIN: u32 = 64;
    pub const HEIGHT_MAX: u32 = 1024;

    pub const GUIDANCE_SCALE_MIN: f64 = 2.5;
    pub const GUIDANCE_SCALE_MAX: f64 = 20.0;

    pub const STEPS_MIN: usize = 5;
    pub const STEPS_MAX: usize = 100;
}

/// misc
pub mod misc {
    /// the factor to scale progress images by to reduce upload size
    pub const PROGRESS_SCALE_FACTOR: u32 = 2;

    /// time in milliseonds to wait between progress updates
    pub const PROGRESS_UPDATE_MS: u64 = 250;

    /// number of models per category
    pub const MODEL_CHUNK_COUNT: usize = 25;
}
