/// Recurrent Neural Networks
use crate::{Device, Kind, Tensor};

// TODO: define some traits for recurrent neural networks.

/// A Long Short-Term Memory (LSTM) neural network.
pub struct LSTM {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
    hidden_dim: i64,
    device: Device,
}

pub struct LSTMState((Tensor, Tensor));

impl LSTM {
    pub fn new(vs: &super::var_store::Path, in_dim: i64, hidden_dim: i64) -> LSTM {
        let gate_dim = 4 * hidden_dim;
        LSTM {
            w_ih: vs.kaiming_uniform("w_ih", &[gate_dim, in_dim]),
            w_hh: vs.kaiming_uniform("w_hh", &[gate_dim, hidden_dim]),
            b_ih: vs.zeros("b_ih", &[gate_dim]),
            b_hh: vs.zeros("b_hh", &[gate_dim]),
            hidden_dim,
            device: vs.device(),
        }
    }

    pub fn zero_state(&self, batch_dim: i64) -> LSTMState {
        let shape = [batch_dim, self.hidden_dim];
        let zeros = Tensor::zeros(&shape, (Kind::Float, self.device));
        LSTMState((zeros.shallow_clone(), zeros.shallow_clone()))
    }

    pub fn step(&self, input: Tensor, in_state: LSTMState) -> LSTMState {
        let LSTMState((h, c)) = in_state;
        let (h, c) = input.lstm_cell(
            &[&h, &c], &self.w_ih, &self.w_hh, Some(&self.b_ih), Some(&self.b_hh));
        LSTMState((h, c))
    }

    pub fn seq(&self, input: Tensor) -> (Tensor, LSTMState) {
        let batch_dim = input.size()[0];
        let shape = [1, batch_dim, self.hidden_dim];
        let zeros = Tensor::zeros(&shape, (Kind::Float, self.device));
        let (output, h, c) = input.lstm(
            &[&zeros, &zeros], 
            &[&self.w_ih, &self.w_hh, &self.b_ih, &self.b_hh],
            /*has_biases=*/true,
            /*num_layers=*/1,
            /*dropout=*/0.,
            /*train=*/false,
            /*bidirectional=*/false,
            /*batch_first=*/true);
        (output, LSTMState((h, c)))
    }
}
