use crate::constant;
use serenity::{builder::CreateApplicationCommand, model::prelude::command::CommandOptionType};
use stable_diffusion_a1111_webui_client as sd;

pub fn populate_generate_options(command: &mut CreateApplicationCommand, models: &[sd::Model]) {
    command
        .create_option(|option| {
            option
                .name(constant::value::PROMPT)
                .description("The prompt to draw")
                .kind(CommandOptionType::String)
                .required(true)
        })
        .create_option(|option| {
            option
                .name(constant::value::NEGATIVE_PROMPT)
                .description("The prompt to avoid drawing")
                .kind(CommandOptionType::String)
                .required(false)
        })
        .create_option(|option| {
            option
                .name(constant::value::SEED)
                .description("The seed to use")
                .kind(CommandOptionType::Integer)
                .required(false)
        })
        .create_option(|option| {
            option
                .name(constant::value::COUNT)
                .description("The number of images to generate")
                .kind(CommandOptionType::Integer)
                .min_int_value(constant::limits::COUNT_MIN)
                .max_int_value(constant::limits::COUNT_MAX)
                .required(false)
        })
        .create_option(|option| {
            option
                .name(constant::value::WIDTH)
                .description("The width of the image")
                .kind(CommandOptionType::Integer)
                .min_int_value(constant::limits::WIDTH_MIN)
                .max_int_value(constant::limits::WIDTH_MAX)
                .required(false)
        })
        .create_option(|option| {
            option
                .name(constant::value::HEIGHT)
                .description("The height of the image")
                .kind(CommandOptionType::Integer)
                .min_int_value(constant::limits::HEIGHT_MIN)
                .max_int_value(constant::limits::HEIGHT_MAX)
                .required(false)
        })
        .create_option(|option| {
            option
                .name(constant::value::GUIDANCE_SCALE)
                .description("The scale of the guidance to apply")
                .kind(CommandOptionType::Number)
                .min_number_value(constant::limits::GUIDANCE_SCALE_MIN)
                .max_number_value(constant::limits::GUIDANCE_SCALE_MAX)
                .required(false)
        })
        .create_option(|option| {
            option
                .name(constant::value::STEPS)
                .description("The number of denoising steps to apply")
                .kind(CommandOptionType::Integer)
                .min_int_value(constant::limits::STEPS_MIN)
                .max_int_value(constant::limits::STEPS_MAX)
                .required(false)
        })
        .create_option(|option| {
            option
                .name(constant::value::TILING)
                .description("Whether or not the image should be tiled at the edges")
                .kind(CommandOptionType::Boolean)
                .required(false)
        })
        .create_option(|option| {
            option
                .name(constant::value::RESTORE_FACES)
                .description("Whether or not the image should have its faces restored")
                .kind(CommandOptionType::Boolean)
                .required(false)
        })
        .create_option(|option| {
            option
                .name(constant::value::DENOISING_STRENGTH)
                .description(
                    "The amount of denoising to apply (0 is no change, 1 is complete remake)",
                )
                .kind(CommandOptionType::Number)
                .min_number_value(0.0)
                .max_number_value(1.0)
                .required(false)
        })
        .create_option(|option| {
            let opt = option
                .name(constant::value::SAMPLER)
                .description("The sampler to use")
                .kind(CommandOptionType::String)
                .required(false);

            for value in sd::Sampler::VALUES {
                opt.add_string_choice(value, value);
            }

            opt
        });

    for (idx, chunk) in models.chunks(constant::misc::MODEL_CHUNK_COUNT).enumerate() {
        command.create_option(|option| {
            let opt = option
                .name(if idx == 0 {
                    constant::value::MODEL.to_string()
                } else {
                    format!("{}{}", constant::value::MODEL, idx + 1)
                })
                .description(format!("The model to use, category {}", idx + 1))
                .kind(CommandOptionType::String)
                .required(false);

            for model in chunk {
                opt.add_string_choice(&model.name, &model.title);
            }

            opt
        });
    }
}
