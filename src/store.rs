use crate::{sd, util};
use anyhow::Context;
use rusqlite::OptionalExtension;
use serenity::model::id::UserId;
use stable_diffusion_a1111_webui_client::Sampler;

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
        format!("`/paint prompt:{}{} seed:{} count:1 width:{} height:{} guidance_scale:{} steps:{} tiling:{} restore_faces:{} sampler:{} {}`",
            self.prompt,
            self.negative_prompt.as_ref().map(|s| format!(" negative_prompt:{s}")).unwrap_or_default(),
            self.seed,
            self.width,
            self.height,
            self.cfg_scale,
            self.steps,
            self.tiling,
            self.restore_faces,
            self.sampler.to_string(),
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

pub struct Store(rusqlite::Connection);
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

            CREATE TABLE IF NOT EXISTS interrogation (
                id	            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id	        TEXT NOT NULL,

                generation_id	INTEGER,
                url	            TEXT,

                result	        TEXT NOT NULL,
                interrogator	TEXT NOT NULL,

                FOREIGN KEY(generation_id)  REFERENCES generation(id)
            ) STRICT;
        ",
            (),
        )?;

        Ok(Self(connection))
    }

    pub fn insert_generation(&mut self, generation: Generation) -> anyhow::Result<i64> {
        let g = generation;
        self.0.execute(
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

        Ok(self.0.last_insert_rowid())
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

    fn get_generation_with_predicate(
        &self,
        predicate: &str,
        params: impl rusqlite::Params,
    ) -> anyhow::Result<Option<Generation>> {
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
        )) = self
            .0
            .query_row::<(_, _, _, _, _, _, _, _, _, String, _, _, String, _), _, _>(
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
