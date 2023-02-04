use crate::{config::Configuration, constant, store::Store, util};
use itertools::Itertools;
use serenity::{
    builder::CreateApplicationCommandOption,
    model::prelude::{
        command::CommandOptionType, interaction::application_command::CommandDataOption, UserId,
    },
};
use stable_diffusion_a1111_webui_client as sd;

pub fn populate_generate_options(
    mut add_option: impl FnMut(CreateApplicationCommandOption),
    models: &[sd::Model],
    with_prompt: bool,
) {
    let limits = &Configuration::get().limits;

    if with_prompt {
        add_option({
            let mut opt = CreateApplicationCommandOption::default();
            opt.name(constant::value::PROMPT)
                .description("The prompt to draw")
                .kind(CommandOptionType::String)
                .required(true);
            opt
        });
    }
    add_option({
        let mut opt = CreateApplicationCommandOption::default();
        opt.name(constant::value::NEGATIVE_PROMPT)
            .description("The prompt to avoid drawing")
            .kind(CommandOptionType::String)
            .required(false);
        opt
    });
    add_option({
        let mut opt = CreateApplicationCommandOption::default();
        opt.name(constant::value::SEED)
            .description("The seed to use")
            .kind(CommandOptionType::Integer)
            .required(false);
        opt
    });
    add_option({
        let mut opt = CreateApplicationCommandOption::default();
        opt.name(constant::value::COUNT)
            .description("The number of images to generate")
            .kind(CommandOptionType::Integer)
            .min_int_value(limits.count_min)
            .max_int_value(limits.count_max)
            .required(false);
        opt
    });
    add_option({
        let mut opt = CreateApplicationCommandOption::default();
        opt.name(constant::value::WIDTH)
            .description("The width of the image")
            .kind(CommandOptionType::Integer)
            .min_int_value(limits.width_min)
            .max_int_value(limits.width_max)
            .required(false);
        opt
    });
    add_option({
        let mut opt = CreateApplicationCommandOption::default();
        opt.name(constant::value::HEIGHT)
            .description("The height of the image")
            .kind(CommandOptionType::Integer)
            .min_int_value(limits.height_min)
            .max_int_value(limits.height_max)
            .required(false);
        opt
    });
    add_option({
        let mut opt = CreateApplicationCommandOption::default();
        opt.name(constant::value::GUIDANCE_SCALE)
            .description("The scale of the guidance to apply")
            .kind(CommandOptionType::Number)
            .min_number_value(limits.guidance_scale_min)
            .max_number_value(limits.guidance_scale_max)
            .required(false);
        opt
    });
    add_option({
        let mut opt = CreateApplicationCommandOption::default();
        opt.name(constant::value::STEPS)
            .description("The number of denoising steps to apply")
            .kind(CommandOptionType::Integer)
            .min_int_value(limits.steps_min)
            .max_int_value(limits.steps_max)
            .required(false);
        opt
    });
    add_option({
        let mut opt = CreateApplicationCommandOption::default();
        opt.name(constant::value::TILING)
            .description("Whether or not the image should be tiled at the edges")
            .kind(CommandOptionType::Boolean)
            .required(false);
        opt
    });
    add_option({
        let mut opt = CreateApplicationCommandOption::default();
        opt.name(constant::value::RESTORE_FACES)
            .description("Whether or not the image should have its faces restored")
            .kind(CommandOptionType::Boolean)
            .required(false);
        opt
    });
    add_option({
        let mut opt = CreateApplicationCommandOption::default();
        opt.name(constant::value::DENOISING_STRENGTH)
            .description("The amount of denoising to apply (0 is no change, 1 is complete remake)")
            .kind(CommandOptionType::Number)
            .min_number_value(0.0)
            .max_number_value(1.0)
            .required(false);
        opt
    });
    add_option({
        let mut opt = CreateApplicationCommandOption::default();
        opt.name(constant::value::SAMPLER)
            .description("The sampler to use")
            .kind(CommandOptionType::String)
            .required(false);

        for value in sd::Sampler::VALUES {
            opt.add_string_choice(value, value);
        }

        opt
    });

    for (idx, chunk) in models
        .chunks(constant::value::MODEL_CHUNK_COUNT)
        .enumerate()
    {
        add_option({
            let mut opt = CreateApplicationCommandOption::default();

            opt.name(if idx == 0 {
                constant::value::MODEL.to_string()
            } else {
                format!("{}{}", constant::value::MODEL, idx + 1)
            })
            .description(format!("The model to use, category {}", idx + 1))
            .kind(CommandOptionType::String)
            .required(false);

            for model in chunk {
                let hash_short = match model.hash_short.as_ref() {
                    Some(hash) => hash,
                    None => panic!("The model '{}' was found without a hash while adding options; this shouldn't be possible. Please report the issue!", model.name),
                };
                opt.add_string_choice(&model.name, hash_short);
            }

            opt
        });
    }
}

#[derive(Clone)]
pub struct OwnedBaseGenerationParameters {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub seed: Option<i64>,
    pub batch_count: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub cfg_scale: Option<f32>,
    pub denoising_strength: Option<f32>,
    pub steps: Option<u32>,
    pub tiling: Option<bool>,
    pub restore_faces: Option<bool>,
    pub sampler: Option<sd::Sampler>,
    pub model: sd::Model,
}
impl OwnedBaseGenerationParameters {
    pub fn load(
        user_id: UserId,
        options: &[CommandDataOption],
        store: &Store,
        models: &[sd::Model],
        use_last_generation_for_size: bool,
        enforce_prompt: bool,
    ) -> anyhow::Result<OwnedBaseGenerationParameters> {
        use util::{
            find_model_by_hash, get_value, get_values_starting_with, value_to_bool, value_to_int,
            value_to_number, value_to_string,
        };

        let prompt = get_value(options, constant::value::PROMPT).and_then(value_to_string);
        let prompt = if let Some(prompt) = prompt {
            prompt
        } else if enforce_prompt {
            anyhow::bail!("expected prompt");
        } else {
            String::new()
        };

        let negative_prompt =
            get_value(options, constant::value::NEGATIVE_PROMPT).and_then(value_to_string);

        let seed = get_value(options, constant::value::SEED).and_then(value_to_int);

        let batch_count = get_value(options, constant::value::COUNT)
            .and_then(value_to_int)
            .map(|v| v as u32);

        let last_generation = store.get_last_generation_for_user(user_id)?;
        let last_generation = last_generation.as_ref();

        let mut width = get_value(options, constant::value::WIDTH)
            .and_then(value_to_int)
            .map(|v| v as u32 / 64 * 64);

        let mut height = get_value(options, constant::value::HEIGHT)
            .and_then(value_to_int)
            .map(|v| v as u32 / 64 * 64);

        if use_last_generation_for_size {
            width = width.or_else(|| last_generation.map(|g| g.width));
            height = height.or_else(|| last_generation.map(|g| g.height));
        }

        let cfg_scale = get_value(options, constant::value::GUIDANCE_SCALE)
            .and_then(value_to_number)
            .map(|v| v as f32)
            .or_else(|| last_generation.map(|g| g.cfg_scale));

        let denoising_strength = get_value(options, constant::value::DENOISING_STRENGTH)
            .and_then(value_to_number)
            .map(|v| v as f32)
            .or_else(|| last_generation.map(|g| g.denoising_strength));

        let steps = get_value(options, constant::value::STEPS)
            .and_then(value_to_int)
            .map(|v| v as u32)
            .or_else(|| last_generation.map(|g| g.steps));

        let tiling = get_value(options, constant::value::TILING)
            .and_then(value_to_bool)
            .or_else(|| last_generation.map(|g| g.tiling));

        let restore_faces = get_value(options, constant::value::RESTORE_FACES)
            .and_then(value_to_bool)
            .or_else(|| last_generation.map(|g| g.restore_faces));

        let sampler = get_value(options, constant::value::SAMPLER)
            .and_then(value_to_string)
            .and_then(|v| sd::Sampler::try_from(v.as_str()).ok())
            .or_else(|| last_generation.map(|g| g.sampler));

        let model = {
            let model_params: Vec<_> = get_values_starting_with(options, constant::value::MODEL)
                .flat_map(value_to_string)
                .collect();
            if model_params.len() > 1 {
                anyhow::bail!(
                    "More than one model was specified: {}",
                    model_params
                        .iter()
                        .map(|hash| util::model_hash_to_name(models, hash.as_str()))
                        .join(", ")
                );
            }

            let model_hash = model_params
                .first()
                .or_else(|| last_generation.map(|g| &g.model_hash));

            let model = model_hash.and_then(|hash| Some(find_model_by_hash(models, hash)?.1));
            match model {
                Some(model) => model,
                None => anyhow::bail!("No model was specified for this request, and you have no past generations to draw upon for a choice of model. Please try again with a model specified."),
            }
        };

        Ok(OwnedBaseGenerationParameters {
            prompt,
            negative_prompt,
            seed,
            batch_count,
            width,
            height,
            cfg_scale,
            denoising_strength,
            steps,
            tiling,
            restore_faces,
            sampler,
            model,
        })
    }

    pub fn as_base_generation_request(&self) -> sd::BaseGenerationRequest {
        sd::BaseGenerationRequest {
            prompt: self.prompt.clone(),
            negative_prompt: self.negative_prompt.clone(),
            seed: self.seed,
            batch_size: Some(1),
            batch_count: self.batch_count,
            width: self.width,
            height: self.height,
            cfg_scale: self.cfg_scale,
            denoising_strength: self.denoising_strength,
            steps: self.steps,
            tiling: self.tiling,
            restore_faces: self.restore_faces,
            sampler: self.sampler,
            model: Some(self.model.clone()),
            ..Default::default()
        }
    }
}
