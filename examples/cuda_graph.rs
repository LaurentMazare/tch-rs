use tch::Tensor;

fn run() -> Result<(), tch::TchError> {
    tch::maybe_init_cuda();
    let mut graph = tch::cuda_graph::CudaGraph::new()?;
    let mut t = Tensor::of_slice(&[3.0]);
    t.print();
    t += 0.1;
    t.print();
    let stream = tch::cuda_stream::CudaStream::get_stream_from_pool(false, 0)?;
    stream.set_current_stream()?;
    graph.capture_begin()?;
    t += 0.01;
    graph.capture_end()?;
    t.print();
    graph.replay()?;
    graph.replay()?;
    graph.replay()?;
    t.print();
    Ok(())
}

fn main() {
    run().unwrap()
}
