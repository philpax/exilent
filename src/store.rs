use crate::{sd, util};
use anyhow::Context;
use itertools::Itertools;
use parking_lot::Mutex;
use rusqlite::OptionalExtension;
use serenity::model::id::UserId;
use stable_diffusion_a1111_webui_client::Sampler;
use std::collections::HashMap;

pub struct Store(Mutex<rusqlite::Connection>);
impl Store {
    const FILENAME: &str = "store.sqlite";

    pub fn load() -> anyhow::Result<Self> {
        let connection = rusqlite::Connection::open(Self::FILENAME)?;
        connection.execute(
            r"
            CREATE TABLE IF NOT EXISTS generation (
                id	                INTEGER PRIMARY KEY AUTOINCREMENT,

                prompt	            TEXT NOT NULL,
                negative_prompt	    TEXT,
                seed	            INTEGER NOT NULL,
                width	            INTEGER NOT NULL,
                height	            INTEGER NOT NULL,
                cfg_scale	        REAL NOT NULL,
                steps	            INTEGER NOT NULL,
                tiling	            INTEGER NOT NULL,
                restore_faces	    INTEGER NOT NULL,
                sampler	            TEXT NOT NULL,
                model_hash	        TEXT NOT NULL,
                image	            BLOB NOT NULL,
                image_url           TEXT,
                denoising_strength  REAL NOT NULL,

                user_id             TEXT NOT NULL,
                timestamp	        TEXT NOT NULL,

                -- img2img specific fields
                init_image          BLOB,
                resize_mode         TEXT,
                init_url            TEXT
            ) STRICT;
            ",
            (),
        )?;
        connection.execute(
            r"
            CREATE TABLE IF NOT EXISTS interrogation (
                id	            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id	        TEXT NOT NULL,
                timestamp	    TEXT NOT NULL,

                generation_id	INTEGER,
                url	            TEXT,

                result	        TEXT NOT NULL,
                interrogator	TEXT NOT NULL,

                FOREIGN KEY(generation_id)  REFERENCES generation(id)
            ) STRICT;
        ",
            (),
        )?;

        Ok(Self(Mutex::new(connection)))
    }

    pub fn insert_generation(&self, generation: Generation) -> anyhow::Result<i64> {
        let g = generation;
        let db = &mut *self.0.lock();
        db.execute(
            r"
            INSERT INTO generation
                (prompt, negative_prompt, seed, width, height, cfg_scale, steps, tiling,
                 restore_faces, sampler, model_hash, image, user_id, timestamp, denoising_strength,
                 init_image, resize_mode, init_url)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ",
            rusqlite::params![
                g.prompt,
                g.negative_prompt,
                g.seed,
                g.width,
                g.height,
                g.cfg_scale,
                g.steps,
                g.tiling,
                g.restore_faces,
                g.sampler.to_string(),
                g.model_hash,
                g.image,
                g.user_id.as_u64().to_string(),
                g.timestamp,
                g.denoising_strength,
                g.image_generation
                    .as_ref()
                    .map(|ig| util::encode_image_to_png_bytes(ig.init_image.clone()))
                    .transpose()?,
                g.image_generation
                    .as_ref()
                    .map(|ig| ig.resize_mode.to_string()),
                g.image_generation.as_ref().map(|ig| ig.init_url.as_str()),
            ],
        )?;

        Ok(db.last_insert_rowid())
    }

    pub fn set_generation_url(&self, key: i64, url: &str) -> anyhow::Result<()> {
        let db = &mut *self.0.lock();
        db.execute(
            r"UPDATE generation SET image_url = ? WHERE id = ?",
            (url, key),
        )?;

        Ok(())
    }

    pub fn get_generation(&self, key: i64) -> anyhow::Result<Option<Generation>> {
        self.get_generation_with_predicate(r"id = ?", [key])
    }

    pub fn get_last_generation_for_user(
        &self,
        user_id: UserId,
    ) -> anyhow::Result<Option<Generation>> {
        self.get_generation_with_predicate(r"user_id = ?", [user_id.as_u64().to_string()])
    }

    pub fn insert_interrogation(&self, interrogation: Interrogation) -> anyhow::Result<i64> {
        let i = interrogation;
        let db = &mut *self.0.lock();
        db.execute(
            r"
            INSERT INTO interrogation
                (user_id, timestamp, generation_id, url, result, interrogator)
            VALUES
                (?, ?, ?, ?, ?, ?)
            ",
            (
                i.user_id.as_u64().to_string(),
                chrono::Local::now(),
                i.source.generation_id(),
                i.source.url(),
                i.result,
                i.interrogator.to_string(),
            ),
        )?;

        Ok(db.last_insert_rowid())
    }

    pub fn get_interrogation(&self, key: i64) -> anyhow::Result<Option<Interrogation>> {
        let db = &mut *self.0.lock();
        let Some((
            user_id, generation_id, url, result, interrogator
        )) = db.query_row(
                r"
                SELECT
                    user_id, generation_id, url, result, interrogator
                FROM
                    interrogation
                WHERE
                    id = ?
                ORDER BY timestamp
                DESC LIMIT 1
                ",
                [key],
                |r| r.try_into(),
            )
            .optional()? else { return Ok(None); };

        Ok(Some(Interrogation::from_db(
            user_id,
            generation_id,
            url,
            result,
            interrogator,
        )?))
    }

    pub fn get_model_usage_counts(&self) -> anyhow::Result<HashMap<UserId, Vec<(String, u64)>>> {
        self.0
            .lock()
            .prepare(
                r#"
                SELECT user_id, model_hash, COUNT(*) AS count
                FROM generation
                GROUP BY user_id, model_hash
                ORDER BY user_id, count DESC
                "#,
            )?
            .query_map([], |row| row.try_into())?
            .flat_map(Result::ok)
            .group_by(|(uid, _, _): &(String, String, i64)| uid.clone())
            .into_iter()
            .map(|(uid, group)| {
                anyhow::Ok((
                    UserId(uid.parse()?),
                    group
                        .into_iter()
                        .map(|(_, hash, count)| (hash, count as u64))
                        .collect::<Vec<(String, u64)>>(),
                ))
            })
            .collect::<Result<_, _>>()
    }
}

#[derive(Debug, Clone)]
pub struct ImageGeneration {
    pub init_image: image::DynamicImage,
    pub init_url: String,
    pub resize_mode: sd::ResizeMode,
}

#[derive(Debug, Clone)]
pub struct Generation {
    pub id: Option<i64>,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub seed: i64,
    pub width: u32,
    pub height: u32,
    pub cfg_scale: f32,
    pub steps: u32,
    pub tiling: bool,
    pub restore_faces: bool,
    pub sampler: Sampler,
    pub model_hash: String,
    pub image: Vec<u8>,
    pub image_url: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Local>,
    pub user_id: UserId,
    pub denoising_strength: f32,
    pub image_generation: Option<ImageGeneration>,
}
impl Generation {
    pub fn as_message(&self, models: &[sd::Model]) -> String {
        use crate::constant as c;
        format!(
            "`/{} {}:{}{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{}{}{}`",
            if self.image_generation.is_some() {
                c::command::PAINTOVER
            } else {
                c::command::PAINT
            },
            c::value::PROMPT,
            self.prompt,
            self.negative_prompt
                .as_ref()
                .map(|s| format!(" {}:{s}", c::value::NEGATIVE_PROMPT))
                .unwrap_or_default(),
            c::value::SEED,
            self.seed,
            c::value::WIDTH,
            self.width,
            c::value::HEIGHT,
            self.height,
            c::value::GUIDANCE_SCALE,
            self.cfg_scale,
            c::value::STEPS,
            self.steps,
            c::value::TILING,
            self.tiling,
            c::value::RESTORE_FACES,
            self.restore_faces,
            c::value::SAMPLER,
            self.sampler,
            c::value::DENOISING_STRENGTH,
            self.denoising_strength,
            util::find_model_by_hash(models, &self.model_hash)
                .map(|(idx, m)| {
                    let model_category = idx / c::misc::MODEL_CHUNK_COUNT;
                    format!(
                        " {}{}:{}",
                        c::value::MODEL,
                        if model_category == 0 {
                            String::new()
                        } else {
                            (model_category + 1).to_string()
                        },
                        m.name
                    )
                })
                .unwrap_or_default(),
            self.image_generation
                .as_ref()
                .map(|ig| {
                    format!(
                        " {}:{} {}:{}",
                        c::value::IMAGE_URL,
                        ig.init_url,
                        c::value::RESIZE_MODE,
                        ig.resize_mode
                    )
                })
                .unwrap_or_default()
        )
    }

    pub fn as_generation_request(&self, models: &[sd::Model]) -> GenerationRequest {
        let base = sd::BaseGenerationRequest {
            prompt: self.prompt.clone(),
            negative_prompt: self.negative_prompt.clone(),
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
            model: util::find_model_by_hash(models, &self.model_hash).map(|t| t.1),
            denoising_strength: Some(self.denoising_strength),
            ..Default::default()
        };

        if let Some(image_generation) = &self.image_generation {
            GenerationRequest::Image(sd::ImageToImageGenerationRequest {
                base,
                resize_mode: Some(image_generation.resize_mode),
                images: vec![image_generation.init_image.clone()],
                ..Default::default()
            })
        } else {
            GenerationRequest::Text(sd::TextToImageGenerationRequest {
                base,
                ..Default::default()
            })
        }
    }
}

pub enum GenerationRequest {
    Text(sd::TextToImageGenerationRequest),
    Image(sd::ImageToImageGenerationRequest),
}
impl GenerationRequest {
    pub fn base(&self) -> &sd::BaseGenerationRequest {
        match self {
            GenerationRequest::Text(r) => &r.base,
            GenerationRequest::Image(r) => &r.base,
        }
    }

    pub fn generate(&self, client: &sd::Client) -> sd::Result<sd::GenerationTask> {
        match self {
            GenerationRequest::Text(r) => client.generate_from_text(r),
            GenerationRequest::Image(r) => client.generate_from_image_and_text(r),
        }
    }
}

#[derive(Debug, Clone)]
pub enum InterrogationSource {
    GenerationId(i64),
    Url(String),
}
impl InterrogationSource {
    pub fn generation_id(&self) -> Option<i64> {
        match self {
            InterrogationSource::GenerationId(id) => Some(*id),
            _ => None,
        }
    }

    pub fn url(&self) -> Option<&str> {
        match self {
            InterrogationSource::Url(url) => Some(url.as_str()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Interrogation {
    pub user_id: UserId,
    pub source: InterrogationSource,
    pub result: String,
    pub interrogator: sd::Interrogator,
}
impl Interrogation {
    pub fn from_db(
        user_id: String,
        generation_id: Option<i64>,
        url: Option<String>,
        result: String,
        interrogator: String,
    ) -> anyhow::Result<Self> {
        let source = match (generation_id, url) {
            (Some(id), None) => InterrogationSource::GenerationId(id),
            (None, Some(url)) => InterrogationSource::Url(url),

            (Some(_), Some(_)) => anyhow::bail!("both generation_id and url were set"),
            (None, None) => anyhow::bail!("neither generation_id or url were set"),
        };

        let interrogator = sd::Interrogator::try_from(interrogator.as_str())
            .ok()
            .context("invalid interrogator")?;

        Ok(Self {
            user_id: UserId(user_id.parse()?),
            source,
            result,
            interrogator,
        })
    }
}

impl Store {
    fn get_generation_with_predicate(
        &self,
        predicate: &str,
        params: impl rusqlite::Params,
    ) -> anyhow::Result<Option<Generation>> {
        let db = &mut *self.0.lock();
        let Some((
            prompt,
            negative_prompt,
            seed,
            width,
            height,
            cfg_scale,
            steps,
            tiling,
            restore_faces,
            sampler,
            model_hash,
            image,
            user_id,
            timestamp,
            denoising_strength,
            init_image,
            resize_mode,
            init_url,
            image_url,
            id,
        )) = db
            .query_row(
                &format!(
                    r"
                    SELECT
                        prompt, negative_prompt, seed, width, height, cfg_scale, steps, tiling,
                        restore_faces, sampler, model_hash, image, user_id, timestamp,
                        denoising_strength, init_image, resize_mode, init_url, image_url, id
                    FROM
                        generation
                    WHERE
                        {}
                    ORDER BY timestamp
                    DESC LIMIT 1
                    ",
                    predicate
                ),
                params,
                |r| {
                    let prompt: String = r.get(0)?;
                    let negative_prompt: Option<String> = r.get(1)?;
                    let seed: i64 = r.get(2)?;
                    let width: u32 = r.get(3)?;
                    let height: u32 = r.get(4)?;
                    let cfg_scale: f32 = r.get(5)?;
                    let steps: u32 = r.get(6)?;
                    let tiling: bool = r.get(7)?;
                    let restore_faces: bool = r.get(8)?;
                    let sampler: String = r.get(9)?;
                    let model_hash: String = r.get(10)?;
                    let image: Vec<u8> = r.get(11)?;
                    let user_id: String = r.get(12)?;
                    let timestamp: chrono::DateTime<chrono::Local> = r.get(13)?;
                    let denoising_strength: f32 = r.get(14)?;
                    let init_image: Option<Vec<u8>> = r.get(15)?;
                    let resize_mode: Option<String> = r.get(16)?;
                    let init_url: Option<String> = r.get(17)?;
                    let image_url: Option<String> = r.get(18)?;
                    let id: i64 = r.get(19)?;

                    Ok((
                        prompt,
                        negative_prompt,
                        seed,
                        width,
                        height,
                        cfg_scale,
                        steps,
                        tiling,
                        restore_faces,
                        sampler,
                        model_hash,
                        image,
                        user_id,
                        timestamp,
                        denoising_strength,
                        init_image,
                        resize_mode,
                        init_url,
                        image_url,
                        id,
                    ))
                },
            )
            .optional()? else { return Ok(None); };

        Ok(Some(Generation {
            id: Some(id),
            prompt,
            seed,
            width,
            height,
            cfg_scale,
            steps,
            tiling,
            restore_faces,
            sampler: Sampler::try_from(sampler.as_str())
                .ok()
                .context("invalid sampler in db")?,
            negative_prompt,
            model_hash,
            image,
            image_url,
            timestamp,
            user_id: UserId(user_id.parse()?),
            denoising_strength,
            image_generation: init_image
                .zip(resize_mode)
                .zip(init_url)
                .map(|((init_image, resize_mode), init_url)| {
                    anyhow::Ok(ImageGeneration {
                        init_image: image::load_from_memory(&init_image)?,
                        init_url,
                        resize_mode: sd::ResizeMode::try_from(resize_mode.as_str())
                            .ok()
                            .context("invalid resize mode")?,
                    })
                })
                .transpose()?,
        }))
    }
}
