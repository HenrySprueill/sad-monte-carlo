//! A test model with fake energy.

use super::*;

use rand::prelude::*;

/// The parameters needed to configure an fake model.
///
/// These parameters are normally set via command-line arguments.
#[derive(Serialize, Deserialize, Debug, AutoArgs)]
#[allow(non_snake_case)]
pub enum Function {
    /// linear
    Linear,
    /// quadratic
    Quadratic {
        /// number of dimensions
        dimensions: usize,
    },
}

impl Function {
    fn dimensions(&self) -> usize {
        match self {
            Function::Linear => 1,
            Function::Quadratic { dimensions } => *dimensions,
        }
    }
    fn energy(&self, r: f64) -> Energy {
        match self {
            Function::Linear => r * Energy::new(1.),
            Function::Quadratic{..} => r*r * Energy::new(1.),
        }
    }
}

/// The parameters needed to configure an fake model.
///
/// These parameters are normally set via command-line arguments.
#[derive(Serialize, Deserialize, Debug, AutoArgs)]
#[allow(non_snake_case)]
pub struct FakeParams {
    /// the function itself
    pub _function: Function,
}

#[allow(non_snake_case)]
/// An Fake model.
#[derive(Serialize, Deserialize, Debug)]
pub struct Fake {
    /// The state of the system
    pub position: Vec<f64>,
    /// The function itself
    pub function: Function,
    /// The last change we made (and might want to undo).
    possible_change: Vec<f64>,
}

impl From<FakeParams> for Fake {
    fn from(parameters: FakeParams) -> Fake {
        Fake {
            position: vec![0.0; parameters._function.dimensions()],
            possible_change: vec![0.0; parameters._function.dimensions()],
            function: parameters._function,
        }
    }
}

impl System for Fake {
    type CollectedData = ();
    fn energy(&self) -> Energy {
        let r = self.position.iter().map(|&x| x * x).sum::<f64>().sqrt();
        self.function.energy(r)
    }
    fn compute_energy(&self) -> Energy {
        self.energy()
    }
}

impl ConfirmSystem for Fake {
    fn confirm(&mut self) {
        self.position = self.possible_change.clone();
    }
}

impl MovableSystem for Fake {
    fn plan_move(&mut self, rng: &mut MyRng, d: Length) -> Option<Energy> {
        let i = rng.gen_range(0, self.position.len());
        self.possible_change = self.position.clone();
        let v: f64 = rng.sample(rand_distr::StandardNormal);
        self.possible_change[i] += v * d.value_unsafe;
        let r = self
            .possible_change
            .iter()
            .map(|&x| x * x)
            .sum::<f64>()
            .sqrt();
        if r > 1. {
            None
        } else {
            Some(self.function.energy(r))
        }
    }
}
