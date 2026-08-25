use tch::{
    inference_mode, is_inference_mode_enabled, no_grad, Device, InferenceModeGuard, Kind, Tensor,
};

#[test]
fn test_inference_mode_basic() {
    // Test that inference mode is initially disabled
    assert!(!is_inference_mode_enabled());

    // Test that inference mode disables gradient tracking
    inference_mode(|| {
        assert!(is_inference_mode_enabled());

        let x = Tensor::randn([10, 10], (Kind::Float, Device::Cpu));
        let x = x.set_requires_grad(true);
        let y = x.matmul(&x);

        // In inference mode, requires_grad is ignored for output
        assert!(!y.requires_grad());
    });

    // Test that inference mode is restored after closure
    assert!(!is_inference_mode_enabled());
}

#[test]
fn test_inference_mode_guard() {
    // Test RAII guard pattern
    assert!(!is_inference_mode_enabled());

    {
        let _guard = InferenceModeGuard::new();
        assert!(is_inference_mode_enabled());
    }

    // Guard is dropped, inference mode should be disabled
    assert!(!is_inference_mode_enabled());
}

#[test]
fn test_inference_mode_guard_default() {
    // Test Default trait implementation
    assert!(!is_inference_mode_enabled());

    {
        let _guard: InferenceModeGuard = Default::default();
        assert!(is_inference_mode_enabled());
    }

    assert!(!is_inference_mode_enabled());
}

#[test]
fn test_nested_inference_mode() {
    // Test nested inference_mode calls
    inference_mode(|| {
        assert!(is_inference_mode_enabled());

        inference_mode(|| {
            // Still enabled in nested call
            assert!(is_inference_mode_enabled());
        });

        // Should still be enabled after nested call returns
        assert!(is_inference_mode_enabled());
    });

    assert!(!is_inference_mode_enabled());
}

#[test]
fn test_inference_mode_with_no_grad() {
    // Test interaction with no_grad

    // Case 1: inference_mode inside no_grad
    no_grad(|| {
        assert!(!is_inference_mode_enabled()); // no_grad doesn't enable inference_mode
        inference_mode(|| {
            assert!(is_inference_mode_enabled());
        });
        // Back to no_grad context, inference_mode should be disabled
        assert!(!is_inference_mode_enabled());
    });

    // Case 2: no_grad inside inference_mode
    inference_mode(|| {
        assert!(is_inference_mode_enabled());
        no_grad(|| {
            // Inference mode remains enabled; no_grad does not disable it
            // The key difference is that inference_mode disables view tracking
            // while no_grad keeps view tracking enabled
            assert!(is_inference_mode_enabled());
        });
        assert!(is_inference_mode_enabled());
    });

    assert!(!is_inference_mode_enabled());
}

#[test]
fn test_inference_mode_preserves_state() {
    // Test that inference mode properly saves and restores previous state

    // Start with inference mode enabled via guard
    let guard = InferenceModeGuard::new();
    assert!(is_inference_mode_enabled());

    // Nested inference_mode should preserve the fact that it was already enabled
    inference_mode(|| {
        assert!(is_inference_mode_enabled());
    });

    // Should still be enabled since we had the guard
    assert!(is_inference_mode_enabled());

    // Drop the guard
    drop(guard);
    assert!(!is_inference_mode_enabled());
}

#[test]
fn test_inference_mode_restores_state_after_panic() {
    assert!(!is_inference_mode_enabled());

    let result = std::panic::catch_unwind(|| {
        inference_mode(|| {
            assert!(is_inference_mode_enabled());
            panic!("intentional panic to verify inference mode cleanup");
        });
    });

    assert!(result.is_err());
    assert!(!is_inference_mode_enabled());
}

#[test]
fn test_inference_mode_computation() {
    // Test that computations actually work in inference mode
    inference_mode(|| {
        let a = Tensor::randn([100, 100], (Kind::Float, Device::Cpu));
        let b = Tensor::randn([100, 100], (Kind::Float, Device::Cpu));

        // Perform various operations
        let c = a.matmul(&b);
        let d = c.relu();
        let e = d.sum(Kind::Float);

        // Results should be valid tensors
        assert_eq!(c.size(), vec![100i64, 100]);
        assert_eq!(d.size(), vec![100i64, 100]);
        assert_eq!(e.size(), Vec::<i64>::new());

        // None should require grad in inference mode
        assert!(!c.requires_grad());
        assert!(!d.requires_grad());
        assert!(!e.requires_grad());
    });
}

#[test]
fn test_inference_mode_thread_safety() {
    use std::thread;

    // Test that inference mode is thread-local
    assert!(!is_inference_mode_enabled());

    let handle = thread::spawn(|| {
        // In new thread, inference mode should be disabled by default
        assert!(!is_inference_mode_enabled());

        inference_mode(|| {
            // Enabled in this thread
            assert!(is_inference_mode_enabled());
        });

        // Disabled again in this thread
        assert!(!is_inference_mode_enabled());
    });

    // Main thread should not be affected
    assert!(!is_inference_mode_enabled());

    handle.join().unwrap();

    // Still disabled in main thread
    assert!(!is_inference_mode_enabled());
}

#[test]
fn test_inference_mode_view_behavior() {
    // Test view operations in inference mode
    inference_mode(|| {
        let x = Tensor::randn([10, 10], (Kind::Float, Device::Cpu));
        let y = x.view([5, 20]);
        let z = y.view([2, 5, 10]);

        // Views should work correctly
        assert_eq!(y.size(), vec![5, 20]);
        assert_eq!(z.size(), vec![2, 5, 10]);

        // In inference mode, views are not tracked as views (they behave like copies)
        // The actual tensor data is shared, but autograd doesn't track the view relationship
        assert!(!y.requires_grad());
        assert!(!z.requires_grad());
    });
}

#[test]
fn test_inference_mode_with_guard_nested() {
    // Test mixing inference_mode closure with guards
    let outer_guard = InferenceModeGuard::new();
    assert!(is_inference_mode_enabled());

    {
        let _inner_guard = InferenceModeGuard::new();
        assert!(is_inference_mode_enabled());
    }

    // Should still be enabled because outer guard exists
    assert!(is_inference_mode_enabled());

    drop(outer_guard);
    assert!(!is_inference_mode_enabled());
}

#[test]
fn test_inference_mode_guards_can_drop_out_of_order() {
    assert!(!is_inference_mode_enabled());

    let outer_guard = InferenceModeGuard::new();
    let inner_guard = InferenceModeGuard::new();
    assert!(is_inference_mode_enabled());

    drop(outer_guard);
    assert!(is_inference_mode_enabled());

    drop(inner_guard);
    assert!(!is_inference_mode_enabled());
}
