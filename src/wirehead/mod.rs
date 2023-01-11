use self::simulation::{FitnessStore, TextGenome};
use crate::command::OwnedBaseGenerationParameters;
use serenity::{http::Http, model::prelude::ChannelId};
use stable_diffusion_a1111_webui_client as sd;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

pub mod command;
pub mod message_component;
mod message_task;
pub mod simulation;

#[derive(Clone)]
pub struct GenerationParameters {
    parameters: OwnedBaseGenerationParameters,
    tags: Vec<String>,
    prefix: Option<String>,
    suffix: Option<String>,
}

pub struct Session {
    _simulation_thread: std::thread::JoinHandle<anyhow::Result<()>>,
    _message_task: tokio::task::JoinHandle<anyhow::Result<()>>,
    fitness_store: Arc<FitnessStore>,
    shutdown: Arc<AtomicBool>,
    hide_prompt: bool,
    generation_parameters: GenerationParameters,
    to_exilent_channel_id: Option<ChannelId>,
    original_message_link: String,
}
impl Session {
    pub fn new(
        http: Arc<Http>,
        channel_id: ChannelId,
        to_exilent_channel_id: Option<ChannelId>,
        client: Arc<sd::Client>,
        hide_prompt: bool,
        generation_parameters: GenerationParameters,
        original_message_link: String,
    ) -> anyhow::Result<Self> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let fitness_store = Arc::new(FitnessStore::new(shutdown.clone()));

        let (result_tx, result_rx) = flume::unbounded();

        let simulation_thread = std::thread::spawn({
            let fitness_store = fitness_store.clone();
            let shutdown = shutdown.clone();
            let tags = generation_parameters.tags.clone();
            move || simulation::thread(fitness_store, shutdown, tags, result_tx)
        });

        let message_task = tokio::task::spawn(message_task::task(message_task::Parameters {
            http,
            channel_id,
            shutdown: shutdown.clone(),
            fitness_store: fitness_store.clone(),
            result_rx,
            to_exilent_enabled: to_exilent_channel_id.is_some(),
            hide_prompt,
            client,
            generation_parameters: generation_parameters.clone(),
        }));

        Ok(Self {
            _simulation_thread: simulation_thread,
            _message_task: message_task,
            fitness_store,
            shutdown,
            hide_prompt,
            generation_parameters,
            to_exilent_channel_id,
            original_message_link,
        })
    }

    pub fn rate(&self, genome: TextGenome, fitness: usize) {
        self.fitness_store.rate(genome, fitness)
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}
