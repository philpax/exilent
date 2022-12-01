use anyhow::Context;

const SEPARATOR: &str = "#";

const GENERATION_PREFIX: &str = "gen";
const INTERROGATION_PREFIX: &str = "int";

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
impl ToString for Generation {
    fn to_string(&self) -> String {
        match self {
            Self::Retry => Self::GENERATION_RETRY,
            Self::RetryWithOptions => Self::GENERATION_RETRY_WITH_OPTIONS,
            Self::RetryWithOptionsResponse => Self::GENERATION_RETRY_WITH_OPTIONS_RESPONSE,
            Self::InterrogateClip => Self::GENERATION_INTERROGATE_CLIP,
            Self::InterrogateDeepDanbooru => Self::GENERATION_INTERROGATE_DEEPDANBOORU,
        }
        .to_string()
    }
}

pub enum Interrogation {}
impl Interrogation {
    #[allow(dead_code)]
    pub fn to_id(self, id: i64) -> CustomId {
        CustomId::Interrogation {
            id,
            interrogation: self,
        }
    }
}
impl TryFrom<&str> for Interrogation {
    type Error = anyhow::Error;

    fn try_from(_value: &str) -> Result<Self, Self::Error> {
        unimplemented!()
    }
}
impl ToString for Interrogation {
    fn to_string(&self) -> String {
        String::new()
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
impl ToString for CustomId {
    fn to_string(&self) -> String {
        match self {
            CustomId::Generation { id, generation } => {
                format!(
                    "{GENERATION_PREFIX}{SEPARATOR}{id}{SEPARATOR}{}",
                    generation.to_string()
                )
            }
            CustomId::Interrogation { id, interrogation } => {
                format!(
                    "{INTERROGATION_PREFIX}{SEPARATOR}{id}{SEPARATOR}{}",
                    interrogation.to_string()
                )
            }
        }
    }
}
