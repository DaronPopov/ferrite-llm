use ferrite_gpu_lang::{GpuLangRuntime, HostTensor, Program};

fn main() -> ferrite_gpu_lang::Result<()> {
    println!("=== CumSum OpCode (0xC0) Test ===\n");

    // Test 1: 1D cumsum [1, 2, 3, 4, 5] -> [1, 3, 6, 10, 15]
    {
        let mut p = Program::new();
        let x = p.input(&[5])?;
        let y = p.cumsum(x, 0);
        p.set_output(y);
        let c = p.compile()?;

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let expected = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let input = HostTensor::new(vec![5], data)?;

        let rt = GpuLangRuntime::new(0)?;
        let out = rt.execute(&c, &[input])?;

        println!("Test 1: 1D cumsum");
        println!("  input:    [1, 2, 3, 4, 5]");
        println!("  output:   {:?}", out.data());
        println!("  expected: {:?}", expected);

        let ok = out.data().iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 1e-5);
        println!("  PASS: {}\n", ok);
        assert!(ok, "1D cumsum failed!");
    }

    // Test 2: 2D cumsum along dim=1 (inner)
    // [[1, 2, 3], [4, 5, 6]] -> [[1, 3, 6], [4, 9, 15]]
    {
        let mut p = Program::new();
        let x = p.input(&[2, 3])?;
        let y = p.cumsum(x, 1);
        p.set_output(y);
        let c = p.compile()?;

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let expected = vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0];
        let input = HostTensor::new(vec![2, 3], data)?;

        let rt = GpuLangRuntime::new(0)?;
        let out = rt.execute(&c, &[input])?;

        println!("Test 2: 2D cumsum along dim=1");
        println!("  input:    [[1,2,3],[4,5,6]]");
        println!("  output:   {:?}", out.data());
        println!("  expected: {:?}", expected);

        let ok = out.data().iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 1e-5);
        println!("  PASS: {}\n", ok);
        assert!(ok, "2D cumsum dim=1 failed!");
    }

    // Test 3: 2D cumsum along dim=0 (outer)
    // [[1, 2, 3], [4, 5, 6]] -> [[1, 2, 3], [5, 7, 9]]
    {
        let mut p = Program::new();
        let x = p.input(&[2, 3])?;
        let y = p.cumsum(x, 0);
        p.set_output(y);
        let c = p.compile()?;

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let expected = vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0];
        let input = HostTensor::new(vec![2, 3], data)?;

        let rt = GpuLangRuntime::new(0)?;
        let out = rt.execute(&c, &[input])?;

        println!("Test 3: 2D cumsum along dim=0");
        println!("  input:    [[1,2,3],[4,5,6]]");
        println!("  output:   {:?}", out.data());
        println!("  expected: {:?}", expected);

        let ok = out.data().iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 1e-5);
        println!("  PASS: {}\n", ok);
        assert!(ok, "2D cumsum dim=0 failed!");
    }

    // Test 4: 3D cumsum along dim=1 (middle)
    // shape [2, 3, 2], scan along dim=1
    {
        let mut p = Program::new();
        let x = p.input(&[2, 3, 2])?;
        let y = p.cumsum(x, 1);
        p.set_output(y);
        let c = p.compile()?;

        // [[[1,2],[3,4],[5,6]], [[7,8],[9,10],[11,12]]]
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        // cumsum along dim=1:
        //   batch 0: [1,2], [1+3,2+4], [1+3+5,2+4+6] = [1,2],[4,6],[9,12]
        //   batch 1: [7,8], [7+9,8+10], [7+9+11,8+10+12] = [7,8],[16,18],[27,30]
        let expected = vec![1.0, 2.0, 4.0, 6.0, 9.0, 12.0, 7.0, 8.0, 16.0, 18.0, 27.0, 30.0];
        let input = HostTensor::new(vec![2, 3, 2], data)?;

        let rt = GpuLangRuntime::new(0)?;
        let out = rt.execute(&c, &[input])?;

        println!("Test 4: 3D cumsum along dim=1 (middle)");
        println!("  output:   {:?}", out.data());
        println!("  expected: {:?}", expected);

        let ok = out.data().iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 1e-5);
        println!("  PASS: {}\n", ok);
        assert!(ok, "3D cumsum dim=1 failed!");
    }

    println!("RESULT mode=test_cumsum");
    println!("RESULT all_tests=PASSED");
    println!("RESULT opcode=0xC0");

    Ok(())
}
