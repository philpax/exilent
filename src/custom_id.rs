use crate::wirehead::simulation::TextGenome;
use anyhow::Context;
use std::fmt::Display;

const SEPARATOR: &str = "#";

const GENERATION_PREFIX: &str = "gen";
const INTERROGATION_PREFIX: &str = "int";
const WIREHEAD_PREFIX: &str = "wh";

macro_rules! implement_custom_id_component {
    ($name:ident, $(($member:ident, $const:ident, $segment:literal)),*) => {
        #[derive(Clone, Copy)]
        pub enum $name {
            $($member,)*
        }
        impl $name {
            $(const $const: &'static str = $segment;)*
        }
        impl TryFrom<&str> for $name {
            type Error = anyhow::Error;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                match value {
                    $(Self::$const => Ok(Self::$member),)*
                    _ => Err(anyhow::anyhow!("invalid command")),
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
impl Generation {
    pub fn to_id(self, id: i64) -> CustomId {
        CustomId::Generation { id, value: self }
    }
}

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
impl Interrogation {
    pub fn to_id(self, id: i64) -> CustomId {
        CustomId::Interrogation { id, value: self }
    }
}

#[derive(Clone, Copy)]
pub struct Wirehead {
    pub value: WireheadValue,
    pub seed: i64,
}
impl Wirehead {
    const SEPARATOR: char = '!';
}
impl TryFrom<&str> for Wirehead {
    type Error = anyhow::Error;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let (prefix, suffix) = value.split_once(Self::SEPARATOR).context("no separator")?;
        Ok(Self {
            value: prefix.try_into()?,
            seed: suffix.parse()?,
        })
    }
}
impl Display for Wirehead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}{}", self.value, Self::SEPARATOR, self.seed)
    }
}

implement_custom_id_component!(
    WireheadValue,
    (Negative2, WIREHEAD_NEGATIVE2, "n2"),
    (Negative1, WIREHEAD_NEGATIVE1, "n1"),
    (Zero, WIREHEAD_ZERO, "z"),
    (Positive1, WIREHEAD_POSITIVE1, "p1"),
    (Positive2, WIREHEAD_POSITIVE2, "p2"),
    (ToExilent, WIREHEAD_TO_EXILENT, "to_exilent")
);
impl WireheadValue {
    pub fn to_id(self, id: TextGenome, seed: i64) -> CustomId {
        CustomId::Wirehead {
            genome: id,
            value: Wirehead { value: self, seed },
        }
    }

    pub fn as_integer(&self) -> i32 {
        match self {
            WireheadValue::Negative2 => -2,
            WireheadValue::Negative1 => -1,
            WireheadValue::Zero => 0,
            WireheadValue::Positive1 => 1,
            WireheadValue::Positive2 => 2,
            WireheadValue::ToExilent { .. } => unreachable!(),
        }
    }
}

pub enum CustomId {
    Generation { id: i64, value: Generation },
    Interrogation { id: i64, value: Interrogation },
    Wirehead { genome: TextGenome, value: Wirehead },
}
impl TryFrom<&str> for CustomId {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let mut split = value.split(SEPARATOR);
        let prefix = split.next().context("missing component")?;
        let id = split.next().context("missing component")?;
        let cmd = split.next().context("missing component")?;

        Ok(match prefix {
            GENERATION_PREFIX => Self::Generation {
                id: id.parse()?,
                value: Generation::try_from(cmd)?,
            },
            INTERROGATION_PREFIX => Self::Interrogation {
                id: id.parse()?,
                value: Interrogation::try_from(cmd)?,
            },
            WIREHEAD_PREFIX => Self::Wirehead {
                genome: hex_to_genome(id),
                value: Wirehead::try_from(cmd)?,
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
                    "{GENERATION_PREFIX}{SEPARATOR}{id}{SEPARATOR}{generation}"
                )
            }
            CustomId::Interrogation {
                id,
                value: interrogation,
            } => {
                write!(
                    f,
                    "{INTERROGATION_PREFIX}{SEPARATOR}{id}{SEPARATOR}{interrogation}"
                )
            }
            CustomId::Wirehead {
                genome: id,
                value: wirehead,
            } => {
                write!(
                    f,
                    "{WIREHEAD_PREFIX}{SEPARATOR}{}{SEPARATOR}{}",
                    genome_to_hex(id.clone()),
                    wirehead
                )
            }
        }
    }
}

fn genome_to_hex(genome: TextGenome) -> String {
    hex::encode(bytemuck::cast_slice::<u16, u8>(genome.as_slice()))
}

fn hex_to_genome(hex: &str) -> TextGenome {
    bytemuck::cast_slice::<u8, u16>(&hex::decode(hex).unwrap())
        .into()
}
