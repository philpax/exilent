use std::sync::Mutex;

use crate::{sd, util};
use anyhow::Context;
use rusqlite::OptionalExtension;
use serenity::model::id::UserId;
use stable_diffusion_a1111_webui_client::Sampler;

pub struct Store(Mutex<rusqlite::Connection>);
impl Store {
    const FILENAME: &str = "store.sqlite";

    pub fn load() -> anyhow::Result<Self> {
        let connection = rusqlite::Connection::open(Self::FILENAME)?;
        connection.execute(
            r"
            CREATE TABLE IF NOT EXISTS generation (
                id	            INTEGER PRIMARY KEY AUTOINCREMENT,

                prompt	        TEXT NOT NULL,
                negative_prompt	TEXT,
                seed	        INTEGER NOT NULL,
                width	        INTEGER NOT NULL,
                height	        INTEGER NOT NULL,
                cfg_scale	    REAL NOT NULL,
                steps	        INTEGER NOT NULL,
                tiling	        INTEGER NOT NULL,
                restore_faces	INTEGER NOT NULL,
                sampler	        TEXT NOT NULL,
                model_hash	    TEXT NOT NULL,
                image	        BLOB NOT NULL,

                user_id         TEXT NOT NULL,
                timestamp	    TEXT NOT NULL
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
        let db = &mut *self.0.lock().unwrap();
        db.execute(
            r"
            INSERT INTO generation
                (prompt, negative_prompt, seed, width, height, cfg_scale, steps, tiling,
                 restore_faces, sampler, model_hash, image, user_id, timestamp)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ",
            (
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
            ),
        )?;

        Ok(db.last_insert_rowid())
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
        let db = &mut *self.0.lock().unwrap();
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
        let db = &mut *self.0.lock().unwrap();
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
}

#[derive(Debug, Clone)]
pub struct Generation {
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
    pub timestamp: chrono::DateTime<chrono::Local>,
    pub user_id: UserId,
}
impl Generation {
    pub fn as_message(&self, models: &[sd::Model]) -> String {
        use crate::constant as c;
        format!(
            "`/{} {}:{}{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{}{}`",
            c::command::PAINT,
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
                .unwrap_or_default()
        )
    }

    pub fn as_generation_request<'a>(
        &'a self,
        models: &'a [sd::Model],
    ) -> sd::GenerationRequest<'a> {
        sd::GenerationRequest {
            prompt: self.prompt.as_str(),
            negative_prompt: self.negative_prompt.as_deref(),
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
            ..Default::default()
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
        let db = &mut *self.0.lock().unwrap();
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
        )) = db.query_row::<(_, _, _, _, _, _, _, _, _, String, _, _, String, _), _, _>(
                &format!(
                    r"
                    SELECT
                        prompt, negative_prompt, seed, width, height, cfg_scale, steps, tiling,
                        restore_faces, sampler, model_hash, image, user_id, timestamp
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
                |r| r.try_into(),
            )
            .optional()? else { return Ok(None); };

        Ok(Some(Generation {
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
            timestamp,
            user_id: UserId(user_id.parse()?),
        }))
    }
}
