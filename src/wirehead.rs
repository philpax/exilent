use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use crate::custom_id as cid;
use genevo::{
    operator::prelude::*,
    population::ValueEncodedGenomeBuilder,
    prelude::*,
    simulation::State,
    termination::{StopFlag, Termination},
};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serenity::{
    http::Http,
    model::prelude::{component::ButtonStyle, ChannelId},
};

use smallvec::SmallVec;
use stable_diffusion_a1111_webui_client as sd;

// genetic config
const TARGET_LEN: usize = 10;
static POPULATION_SIZE: Lazy<usize> = Lazy::new(|| (10. * (TARGET_LEN as f64).ln()) as usize);
static NUM_INDIVIDUALS_PER_PARENTS: Lazy<usize> = Lazy::new(|| 3);
static SELECTION_RATIO: Lazy<f64> = Lazy::new(|| 0.7);
static NUM_CROSSOVER_POINTS: Lazy<usize> = Lazy::new(|| TARGET_LEN / 6);
static MUTATION_RATE: Lazy<f64> = Lazy::new(|| 0.05 / (TARGET_LEN as f64).ln());
static REINSERTION_RATIO: Lazy<f64> = Lazy::new(|| 0.7);

// stable diffusion config
const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const CFG_SCALE: f32 = 8.0;
const SAMPLER: sd::Sampler = sd::Sampler::EulerA;
const STEPS: u32 = 15;
pub const MODEL_NAME_PREFIX: &str = "Anything";

const TIME_BETWEEN_BLOCK_CHECKS: u64 = 100;
const GENERATION_FAILED_IMAGE: &'static [u8] = include_bytes!("generation-failed.png");

/// The phenotype
pub type Text = String;

/// The genotype
pub type TextGenome = SmallVec<[u16; TARGET_LEN]>;

/// How do the genes of the genotype show up in the phenotype
pub trait AsPhenotype {
    fn as_text(&self, tags: &[String]) -> Text;
}

impl AsPhenotype for TextGenome {
    fn as_text(&self, tags: &[String]) -> Text {
        self.iter()
            .map(|i| tags[*i as usize].as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

pub struct Session {
    _simulation_thread: std::thread::JoinHandle<()>,
    _message_thread: tokio::task::JoinHandle<()>,
    fitness_store: Arc<FitnessStore>,
    shutdown: Arc<AtomicBool>,
    pub tags: Vec<String>,
}
impl Session {
    pub fn new(
        channel_id: ChannelId,
        http: Arc<Http>,
        client: Arc<sd::Client>,
        model: Arc<sd::Model>,
        tags: Vec<String>,
    ) -> anyhow::Result<Self> {
        let min_value = 0;
        let max_value = u16::try_from(tags.len())?;

        let initial_population: Population<TextGenome> = build_population()
            .with_genome_builder(ValueEncodedGenomeBuilder::new(
                TARGET_LEN, min_value, max_value,
            ))
            .of_size(*POPULATION_SIZE)
            .uniform_at_random();

        let shutdown = Arc::new(AtomicBool::new(false));
        let fitness_store = Arc::new(FitnessStore::new(shutdown.clone()));

        struct NeverTerminate;
        impl<A: Algorithm> Termination<A> for NeverTerminate {
            fn evaluate(&mut self, _state: &State<A>) -> StopFlag {
                StopFlag::Continue
            }
        }

        let (result_tx, result_rx) = flume::unbounded();

        let simulation_thread = std::thread::spawn({
            let fitness_store = fitness_store.clone();
            let shutdown = shutdown.clone();
            move || {
                let mut simulator = simulate(
                    genetic_algorithm()
                        .with_evaluation(FitnessCalc {
                            store: fitness_store.clone(),
                        })
                        .with_selection(MaximizeSelector::new(
                            *SELECTION_RATIO,
                            *NUM_INDIVIDUALS_PER_PARENTS,
                        ))
                        .with_crossover(MultiPointCrossBreeder::new(*NUM_CROSSOVER_POINTS))
                        .with_mutation(RandomValueMutator::new(
                            *MUTATION_RATE,
                            min_value,
                            max_value,
                        ))
                        .with_reinsertion(ElitistReinserter::new(
                            FitnessCalc {
                                store: fitness_store.clone(),
                            },
                            true,
                            *REINSERTION_RATIO,
                        ))
                        .with_initial_population(initial_population)
                        .build(),
                )
                .until(NeverTerminate)
                .build();

                loop {
                    let result = simulator.step();
                    if shutdown.load(Ordering::SeqCst) {
                        break;
                    }

                    match result {
                        Ok(SimResult::Intermediate(step)) => {
                            result_tx
                                .send(step.result.best_solution.solution.genome.clone())
                                .unwrap();
                        }
                        Ok(SimResult::Final(..)) => {
                            break;
                        }
                        Err(error) => {
                            println!("{}", error);
                            break;
                        }
                    }
                }
            }
        });

        let message_task = tokio::task::spawn({
            async fn generate(
                client: &sd::Client,
                model: sd::Model,
                prompt: String,
            ) -> anyhow::Result<Vec<u8>> {
                pub fn encode_image_to_png_bytes(
                    image: &image::DynamicImage,
                ) -> anyhow::Result<Vec<u8>> {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut cursor = std::io::Cursor::new(&mut bytes);
                    image.write_to(&mut cursor, image::ImageOutputFormat::Png)?;
                    Ok(bytes)
                }

                let result = client
                    .generate_from_text(&sd::TextToImageGenerationRequest {
                        base: sd::BaseGenerationRequest {
                            prompt,
                            negative_prompt: None,
                            batch_size: Some(1),
                            batch_count: Some(1),
                            width: Some(WIDTH),
                            height: Some(HEIGHT),
                            cfg_scale: Some(CFG_SCALE),
                            denoising_strength: None,
                            eta: None,
                            sampler: Some(SAMPLER),
                            steps: Some(STEPS),
                            model: Some(model),
                            ..Default::default()
                        },
                        ..Default::default()
                    })?
                    .block()
                    .await;

                let image = match result {
                    Ok(result) => result.images[0].clone(),
                    Err(err) => {
                        println!("generation failed: {:?}", err);
                        image::load_from_memory(GENERATION_FAILED_IMAGE)?
                    }
                };

                encode_image_to_png_bytes(&image)
            }

            let fitness_store = fitness_store.clone();
            let shutdown = shutdown.clone();
            let tags = tags.clone();
            async move {
                loop {
                    if shutdown.load(Ordering::SeqCst) {
                        break;
                    }

                    if let Ok(result) = result_rx.try_recv() {
                        let image = generate(&client, (*model).clone(), result.as_text(&tags))
                            .await
                            .unwrap();

                        channel_id
                            .send_files(http.as_ref(), [(image.as_slice(), "output.png")], |m| {
                                m.content(format!(
                                    "**Best result so far**: `{}`",
                                    result.as_text(&tags)
                                ))
                            })
                            .await
                            .unwrap();
                    }

                    let pending_requests =
                        std::mem::take(&mut *fitness_store.pending_requests.lock());

                    for genome in pending_requests {
                        let image = generate(&client, (*model).clone(), genome.as_text(&tags))
                            .await
                            .unwrap();

                        channel_id
                            .send_files(http.as_ref(), [(image.as_slice(), "output.png")], |m| {
                                m.content(format!("`{}`", genome.as_text(&tags)))
                                    .components(|mc| {
                                        mc.create_action_row(|r| {
                                            r.create_button(|b| {
                                                b.custom_id(
                                                    cid::Wirehead::Negative2.to_id(genome.clone()),
                                                )
                                                .label("-2")
                                                .style(ButtonStyle::Danger)
                                            })
                                            .create_button(|b| {
                                                b.custom_id(
                                                    cid::Wirehead::Negative1.to_id(genome.clone()),
                                                )
                                                .label("-1")
                                                .style(ButtonStyle::Danger)
                                            })
                                            .create_button(|b| {
                                                b.custom_id(
                                                    cid::Wirehead::Zero.to_id(genome.clone()),
                                                )
                                                .label("0")
                                                .style(ButtonStyle::Secondary)
                                            })
                                            .create_button(|b| {
                                                b.custom_id(
                                                    cid::Wirehead::Positive1.to_id(genome.clone()),
                                                )
                                                .label("1")
                                                .style(ButtonStyle::Success)
                                            })
                                            .create_button(|b| {
                                                b.custom_id(
                                                    cid::Wirehead::Positive2.to_id(genome.clone()),
                                                )
                                                .label("2")
                                                .style(ButtonStyle::Success)
                                            })
                                        })
                                    })
                            })
                            .await
                            .unwrap();
                    }

                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
            }
        });

        Ok(Self {
            _simulation_thread: simulation_thread,
            _message_thread: message_task,
            fitness_store,
            shutdown,
            tags,
        })
    }

    pub fn rate(&self, genome: TextGenome, fitness: usize) {
        self.fitness_store
            .store
            .lock()
            .insert(genome, Score::Ready(fitness));
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

#[derive(Debug, Copy, Clone)]
enum Score {
    Requested,
    Ready(usize),
}

#[derive(Debug)]
struct FitnessStore {
    store: Mutex<HashMap<TextGenome, Score>>,
    pending_requests: Mutex<HashSet<TextGenome>>,
    shutdown: Arc<AtomicBool>,
}
impl FitnessStore {
    fn new(shutdown: Arc<AtomicBool>) -> Self {
        Self {
            store: Mutex::new(HashMap::new()),
            pending_requests: Mutex::new(HashSet::new()),
            shutdown,
        }
    }

    fn block_on_result(&self, genome: &TextGenome) -> usize {
        loop {
            if self.shutdown.load(Ordering::SeqCst) {
                return 0;
            }

            let value = self.store.lock().get(genome).cloned();
            if let Some(score) = value {
                if let Score::Ready(score) = score {
                    return score;
                }
            } else {
                self.store.lock().insert(genome.clone(), Score::Requested);
                self.pending_requests.lock().insert(genome.clone());
            }

            std::thread::sleep(std::time::Duration::from_millis(TIME_BETWEEN_BLOCK_CHECKS));
        }
    }
}

#[derive(Clone, Debug)]
struct FitnessCalc {
    store: Arc<FitnessStore>,
}
impl FitnessCalc {
    const HIGHEST_POSSIBLE_FITNESS: usize = 100;
    const LOWEST_POSSIBLE_FITNESS: usize = 0;
}
impl FitnessFunction<TextGenome, usize> for FitnessCalc {
    fn fitness_of(&self, genome: &TextGenome) -> usize {
        self.store.block_on_result(genome)
    }

    fn average(&self, fitness_values: &[usize]) -> usize {
        fitness_values.iter().sum::<usize>() / fitness_values.len()
    }

    fn highest_possible_fitness(&self) -> usize {
        Self::HIGHEST_POSSIBLE_FITNESS
    }

    fn lowest_possible_fitness(&self) -> usize {
        Self::LOWEST_POSSIBLE_FITNESS
    }
}
