use anyhow::Context;
use std::fmt::Display;

const SEPARATOR: &str = "#";

const GENERATION_PREFIX: &str = "gen";
const INTERROGATION_PREFIX: &str = "int";

#[derive(Clone, Copy)]
pub enum Generation {
    Retry,
    RetryWithOptions,
    RetryWithOptionsResponse,
    InterrogateClip,
    InterrogateDeepDanbooru,
}
impl Generation {
    const GENERATION_RETRY: &str = "retry";
    const GENERATION_RETRY_WITH_OPTIONS: &str = "retry_with_options";
    const GENERATION_RETRY_WITH_OPTIONS_RESPONSE: &str = "retry_with_options_response";
    const GENERATION_INTERROGATE_CLIP: &str = "interrogate_clip";
    const GENERATION_INTERROGATE_DEEPDANBOORU: &str = "interrogate_dd";

    pub fn to_id(self, id: i64) -> CustomId {
        CustomId::Generation {
            id,
            generation: self,
        }
    }
}
impl TryFrom<&str> for Generation {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            Self::GENERATION_RETRY => Ok(Self::Retry),
            Self::GENERATION_RETRY_WITH_OPTIONS => Ok(Self::RetryWithOptions),
            Self::GENERATION_RETRY_WITH_OPTIONS_RESPONSE => Ok(Self::RetryWithOptionsResponse),
            Self::GENERATION_INTERROGATE_CLIP => Ok(Self::InterrogateClip),
            Self::GENERATION_INTERROGATE_DEEPDANBOORU => Ok(Self::InterrogateDeepDanbooru),
            _ => Err(anyhow::anyhow!("invalid command for generation")),
        }
    }
}
impl Display for Generation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Retry => Self::GENERATION_RETRY,
                Self::RetryWithOptions => Self::GENERATION_RETRY_WITH_OPTIONS,
                Self::RetryWithOptionsResponse => Self::GENERATION_RETRY_WITH_OPTIONS_RESPONSE,
                Self::InterrogateClip => Self::GENERATION_INTERROGATE_CLIP,
                Self::InterrogateDeepDanbooru => Self::GENERATION_INTERROGATE_DEEPDANBOORU,
            }
        )
    }
}

#[derive(Clone, Copy)]
pub enum Interrogation {
    Generate,
    ReinterrogateWithClip,
    ReinterrogateWithDeepDanbooru,
}
impl Interrogation {
    const INTERROGATION_GENERATE: &str = "generate";
    const INTERROGATION_REINTERROGATE_CLIP: &str = "reint_clip";
    const INTERROGATION_REINTERROGATE_DD: &str = "reint_dd";

    pub fn to_id(self, id: i64) -> CustomId {
        CustomId::Interrogation {
            id,
            interrogation: self,
        }
    }
}
impl TryFrom<&str> for Interrogation {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            Self::INTERROGATION_GENERATE => Ok(Self::Generate),
            Self::INTERROGATION_REINTERROGATE_CLIP => Ok(Self::ReinterrogateWithClip),
            Self::INTERROGATION_REINTERROGATE_DD => Ok(Self::ReinterrogateWithDeepDanbooru),
            _ => Err(anyhow::anyhow!("invalid command for interrogation")),
        }
    }
}
impl Display for Interrogation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Generate => Self::INTERROGATION_GENERATE,
                Self::ReinterrogateWithClip => Self::INTERROGATION_REINTERROGATE_CLIP,
                Self::ReinterrogateWithDeepDanbooru => Self::INTERROGATION_REINTERROGATE_DD,
            }
        )
    }
}

pub enum CustomId {
    Generation {
        id: i64,
        generation: Generation,
    },
    Interrogation {
        id: i64,
        interrogation: Interrogation,
    },
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
                generation: Generation::try_from(cmd)?,
            },
            INTERROGATION_PREFIX => Self::Interrogation {
                id,
                interrogation: Interrogation::try_from(cmd)?,
            },
            _ => anyhow::bail!("invalid custom id prefix: {prefix}"),
        })
    }
}
impl Display for CustomId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CustomId::Generation { id, generation } => {
                write!(
                    f,
                    "{GENERATION_PREFIX}{SEPARATOR}{id}{SEPARATOR}{}",
                    generation
                )
            }
            CustomId::Interrogation { id, interrogation } => {
                write!(
                    f,
                    "{INTERROGATION_PREFIX}{SEPARATOR}{id}{SEPARATOR}{}",
                    interrogation
                )
            }
        }
    }
}
