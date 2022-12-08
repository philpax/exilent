use anyhow::Context;
use std::fmt::Display;

const SEPARATOR: &str = "#";

const GENERATION_PREFIX: &str = "gen";
const INTERROGATION_PREFIX: &str = "int";

macro_rules! implement_custom_id_component {
    ($name:ident, $(($member:ident, $const:ident, $segment:literal)),*) => {
        #[derive(Clone, Copy)]
        pub enum $name {
            $($member,)*
        }
        impl $name {
            $(const $const: &str = $segment;)*

            pub fn to_id(self, id: i64) -> CustomId {
                CustomId::$name {
                    id,
                    value: self,
                }
            }
        }
        impl TryFrom<&str> for $name {
            type Error = anyhow::Error;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                match value {
                    $(Self::$const => Ok(Self::$member),)*
                    _ => Err(anyhow::anyhow!("invalid command for generation")),
                }
            }
        }
        impl Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(
                    f,
                    "{}",
                    match self {
                        $(Self::$member => Self::$const,)*
                    }
                )
            }
        }
    };
}

implement_custom_id_component!(
    Generation,
    (Retry, GENERATION_RETRY, "retry"),
    (
        RetryWithOptions,
        GENERATION_RETRY_WITH_OPTIONS,
        "retry_with_options"
    ),
    (
        RetryWithOptionsResponse,
        GENERATION_RETRY_WITH_OPTIONS_RESPONSE,
        "retry_with_options_response"
    ),
    (Remix, GENERATION_REMIX, "remix"),
    (RemixResponse, GENERATION_REMIX_RESPONSE, "remix_response"),
    (Upscale, GENERATION_UPSCALE, "upscale"),
    (
        InterrogateClip,
        GENERATION_INTERROGATE_CLIP,
        "interrogate_clip"
    ),
    (
        InterrogateDeepDanbooru,
        GENERATION_INTERROGATE_DEEPDANBOORU,
        "interrogate_dd"
    )
);

implement_custom_id_component!(
    Interrogation,
    (Generate, INTERROGATION_GENERATE, "generate"),
    (
        ReinterrogateWithClip,
        INTERROGATION_REINTERROGATE_CLIP,
        "reint_clip"
    ),
    (
        ReinterrogateWithDeepDanbooru,
        INTERROGATION_REINTERROGATE_DD,
        "reint_dd"
    )
);

pub enum CustomId {
    Generation { id: i64, value: Generation },
    Interrogation { id: i64, value: Interrogation },
}
impl TryFrom<&str> for CustomId {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let mut split = value.split(SEPARATOR);
        let prefix = split.next().context("missing component")?;
        let id = split.next().context("missing component")?.parse::<i64>()?;
        let cmd = split.next().context("missing component")?;

        Ok(match prefix {
            GENERATION_PREFIX => Self::Generation {
                id,
                value: Generation::try_from(cmd)?,
            },
            INTERROGATION_PREFIX => Self::Interrogation {
                id,
                value: Interrogation::try_from(cmd)?,
            },
            _ => anyhow::bail!("invalid custom id prefix: {prefix}"),
        })
    }
}
impl Display for CustomId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CustomId::Generation {
                id,
                value: generation,
            } => {
                write!(
                    f,
                    "{GENERATION_PREFIX}{SEPARATOR}{id}{SEPARATOR}{}",
                    generation
                )
            }
            CustomId::Interrogation {
                id,
                value: interrogation,
            } => {
                write!(
                    f,
                    "{INTERROGATION_PREFIX}{SEPARATOR}{id}{SEPARATOR}{}",
                    interrogation
                )
            }
        }
    }
}
