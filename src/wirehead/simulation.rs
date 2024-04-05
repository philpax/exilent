use genevo::{
    operator::prelude::*,
    population::ValueEncodedGenomeBuilder,
    prelude::*,
    simulation::State,
    termination::{StopFlag, Termination},
};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

/// The phenotype
pub type Text = String;

/// The genotype
pub type TextGenome = SmallVec<[u16; TARGET_LEN]>;

// genetic config
const TARGET_LEN: usize = 10;
static POPULATION_SIZE: Lazy<usize> = Lazy::new(|| (10. * (TARGET_LEN as f64).ln()) as usize);
static NUM_INDIVIDUALS_PER_PARENTS: Lazy<usize> = Lazy::new(|| 3);
static SELECTION_RATIO: Lazy<f64> = Lazy::new(|| 0.7);
static NUM_CROSSOVER_POINTS: Lazy<usize> = Lazy::new(|| TARGET_LEN / 6);
static MUTATION_RATE: Lazy<f64> = Lazy::new(|| 0.05 / (TARGET_LEN as f64).ln());
static REINSERTION_RATIO: Lazy<f64> = Lazy::new(|| 0.7);

const TIME_BETWEEN_BLOCK_CHECKS: u64 = 100;

/// How do the genes of the genotype show up in the phenotype
pub trait AsPhenotype {
    fn as_text(&self, tags: &[String], prefix: Option<&str>, suffix: Option<&str>) -> Text;
}

impl AsPhenotype for TextGenome {
    fn as_text(&self, tags: &[String], prefix: Option<&str>, suffix: Option<&str>) -> Text {
        prefix
            .into_iter()
            .chain(self.iter().map(|i| tags[*i as usize].as_str()))
            .chain(suffix)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

pub fn thread(
    fitness_store: Arc<FitnessStore>,
    shutdown: Arc<AtomicBool>,
    tags: Vec<String>,
    result_tx: flume::Sender<TextGenome>,
) -> anyhow::Result<()> {
    struct NeverTerminate;
    impl<A: Algorithm> Termination<A> for NeverTerminate {
        fn evaluate(&mut self, _state: &State<A>) -> StopFlag {
            StopFlag::Continue
        }
    }

    let min_value = 0;
    let max_value = u16::try_from(tags.len())?;

    let initial_population: Population<TextGenome> = build_population()
        .with_genome_builder(ValueEncodedGenomeBuilder::new(
            TARGET_LEN, min_value, max_value,
        ))
        .of_size(*POPULATION_SIZE)
        .uniform_at_random();

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
                    store: fitness_store,
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
                result_tx.send(step.result.best_solution.solution.genome.clone())?;
            }
            Ok(SimResult::Final(..)) => {
                break;
            }
            Err(error) => {
                println!("{error}");
                break;
            }
        }
    }

    Ok(())
}

#[derive(Debug, Copy, Clone)]
pub enum Score {
    Requested,
    Ready(usize),
}

#[derive(Debug)]
pub struct FitnessStore {
    store: Mutex<HashMap<TextGenome, Score>>,
    pub pending_requests: Mutex<HashSet<TextGenome>>,
    shutdown: Arc<AtomicBool>,
}
impl FitnessStore {
    pub fn new(shutdown: Arc<AtomicBool>) -> Self {
        Self {
            store: Mutex::new(HashMap::new()),
            pending_requests: Mutex::new(HashSet::new()),
            shutdown,
        }
    }

    pub fn rate(&self, genome: TextGenome, fitness: usize) {
        self.store.lock().insert(genome, Score::Ready(fitness));
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
