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

pub struct Session {
    _simulation_thread: std::thread::JoinHandle<anyhow::Result<()>>,
    _message_task: tokio::task::JoinHandle<anyhow::Result<()>>,
    fitness_store: Arc<FitnessStore>,
    shutdown: Arc<AtomicBool>,
    pub tags: Vec<String>,
}
impl Session {
    pub fn new(
        http: Arc<Http>,
        channel_id: ChannelId,
        client: Arc<sd::Client>,
        params: OwnedBaseGenerationParameters,
        tags: Vec<String>,
    ) -> anyhow::Result<Self> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let fitness_store = Arc::new(FitnessStore::new(shutdown.clone()));

        let (result_tx, result_rx) = flume::unbounded();

        let simulation_thread = std::thread::spawn({
            let fitness_store = fitness_store.clone();
            let shutdown = shutdown.clone();
            let tags = tags.clone();
            move || simulation::thread(fitness_store, shutdown, tags, result_tx)
        });

        let message_task = tokio::task::spawn(message_task::task(
            http,
            channel_id,
            client,
            params,
            fitness_store.clone(),
            shutdown.clone(),
            tags.clone(),
            result_rx,
        ));

        Ok(Self {
            _simulation_thread: simulation_thread,
            _message_task: message_task,
            fitness_store,
            shutdown,
            tags,
        })
    }

    pub fn rate(&self, genome: TextGenome, fitness: usize) {
        self.fitness_store.rate(genome, fitness)
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}
