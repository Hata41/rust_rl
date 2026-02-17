use rustpool::core::types::{ArrayData, GenericObs};

pub trait BufferStorage {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub fn flatten_obs(obs: &GenericObs) -> Vec<f32> {
    let capacity = obs
        .iter()
        .map(|a| match a {
            ArrayData::Float32(v) => v.len(),
            ArrayData::Int32(v) => v.len(),
            ArrayData::Bool(v) => v.len(),
        })
        .sum();
    let mut out = Vec::with_capacity(capacity);
    flatten_obs_extend(obs, &mut out);
    out
}

pub fn flatten_obs_into(obs: &GenericObs, out: &mut [f32]) {
    let mut cursor = 0usize;
    for a in obs {
        match a {
            ArrayData::Float32(v) => {
                let end = cursor + v.len();
                out[cursor..end].copy_from_slice(v);
                cursor = end;
            }
            ArrayData::Int32(v) => {
                for &x in v {
                    out[cursor] = x as f32;
                    cursor += 1;
                }
            }
            ArrayData::Bool(v) => {
                for &b in v {
                    out[cursor] = if b { 1.0 } else { 0.0 };
                    cursor += 1;
                }
            }
        }
    }
    debug_assert_eq!(cursor, out.len());
}

pub fn flatten_obs_nonempty(obs: &GenericObs) -> anyhow::Result<Vec<f32>> {
    let out = flatten_obs(obs);
    if out.is_empty() {
        anyhow::bail!("observation flatten produced empty vector");
    }
    Ok(out)
}

fn flatten_obs_extend(obs: &GenericObs, out: &mut Vec<f32>) {
    for a in obs {
        match a {
            ArrayData::Float32(v) => out.extend_from_slice(v),
            ArrayData::Int32(v) => out.extend(v.iter().map(|x| *x as f32)),
            ArrayData::Bool(v) => out.extend(v.iter().map(|b| if *b { 1.0 } else { 0.0 })),
        }
    }
}
