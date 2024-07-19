use cubecl::prelude::*;

// TODO this should probably use householder reflections, meant to be more numerically stable.
#[cube(launch)]
fn gram_schmidt<F: Float>(input: &Array<F>, v: &mut Array<F>, output1: &mut Array<F>, output2: &mut Array<F>) {
    // Parallelise across different matrices. Loop within each case of the QR algorithm,
    //  corresponding to 1 workgroup.
    if UNIT_POS_X > 0 || UNIT_POS_Y > 0 {
        return;
    }

    // Stop WGPU dropping arrays unused
    input[0];
    v[0];
    output1[0];
    output2[0];

    let x = input;
    let q = output1;
    let r = output2;

    /*
        // "Modified Gram-Schmidt"
        https://www.math.uci.edu/~ttrogdon/105A/html/Lecture23.html

        for  j=1:n
            vj=xj
        endfor

        for  j=1:n
            qj=vj/∥vj∥2
            for  k=j+1:n
                vk=vk−(qTjvk)qj
            endfor
        endfor
    */

    // This applies across rows and columns, for each matrix.
    for j in range(0, CUBE_DIM_X * CUBE_DIM_Y, Comptime::new(false)) {
        v[CUBE_POS + j] = x[CUBE_POS + j];
    }

    for j in range(0, CUBE_DIM_Y, Comptime::new(false)) {
        // Calculate ∥vj∥2
        // TODO better norm
        let mut vj_norm_squared: F = F::new(0.0);
        for i in range(0, CUBE_DIM_X, Comptime::new(false)) {
            let v_ji = v[CUBE_POS + j * CUBE_DIM_X + i];
            // F::powf is breaking inside for loops?
            // let l2_norm_squared_i: F = F::powf(v_ji, 2.0.into());
            vj_norm_squared += v_ji * v_ji;
        }
        let vj_norm = F::sqrt(vj_norm_squared);
        
        // Diagonal R values
        r[CUBE_POS + CUBE_DIM_Y * j + j] = vj_norm;

        // Set qj=vj/∥vj∥2
        for i in range(0, CUBE_DIM_X, Comptime::new(false)) {
            q[CUBE_POS + CUBE_DIM_X * j + i] = v[CUBE_POS + j * CUBE_DIM_X + i] / vj_norm;
        }

        for k in range(j + 1, CUBE_DIM_Y, Comptime::new(false)) {
            // vk=vk−(qTjvk)qj

            // TODO better inner product / dot product / (use matrix multiply?)
            let mut qj_dot_vk: F = F::new(0.0);
            for i in range(0, CUBE_DIM_X, Comptime::new(false)) {
                qj_dot_vk += q[CUBE_POS + CUBE_DIM_X * j + i] * v[CUBE_POS + k * CUBE_DIM_X + i];
            }

            // Off-diagonal R values
            r[CUBE_POS + CUBE_DIM_Y * k + j] = qj_dot_vk;

            for i in range(0, CUBE_DIM_X, Comptime::new(false)) {
                v[CUBE_POS + k * CUBE_DIM_X + i] +=
                    F::new(1.0 - 2.0) * qj_dot_vk * q[CUBE_POS + CUBE_DIM_X * j + i];
            }
        }
    }
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    // TODO only works for CubeCount = 1, i.e. single workgroup at the moment.
    //  Some bug overlap.
    let input_shape = [1, 3, 5];
    let input = &[
        -1.0, 0.0, 3.0, 1.0, 5.0, 2.0, 5.0, 1.0, -1.0, 0.0, 1.0, -5.0, 6.0, 3.0, 1.0,
    ];

    // let input_shape = [1, 3, 3];
    // let input = &[1., 1., 0., 1., 0., 1., 0., 1., 1.];
    let [num_matrices, height, width] = input_shape;

    let output_1_handle = client.empty(input.len() * core::mem::size_of::<f32>());
    let output_2_handle = client.empty((height * height) * core::mem::size_of::<f32>());
    let v_handle = client.empty(input.len() * width * core::mem::size_of::<f32>());

    gram_schmidt::launch::<F32, R>(
        &client,
        CubeCount::Static(num_matrices as u32, 1, 1),
        CubeDim::new(width as u32, height as u32, 1),
        ArrayArg::new(&client.create(f32::as_bytes(input)), input.len()),
        ArrayArg::new(&v_handle, num_matrices * width),
        ArrayArg::new(&output_1_handle, input.len()),
        ArrayArg::new(&output_2_handle, input.len()),
    );

    let bytes_1 = client.read(output_1_handle.binding());
    let output_1 = f32::from_bytes(&bytes_1);
    
    let bytes_2 = client.read(output_2_handle.binding());
    let output_2 = f32::from_bytes(&bytes_2);
    
    println!("Q {:?} => {output_1:?}", R::name());
    println!("R {:?} => {output_2:?}", R::name());
    // Check with numpy
    /*
        import numpy as np
        x = [
            [-1.0, 0.0, 3.0, 1.0, 5.0],
            [2.0, 5.0, 1.0, -1.0, 0.0],
            [1.0, -5.0, 6.0, 3.0, 1.0],
        ]
        x = np.array(x).T
        qr = np.linalg.qr(x)
        print(qr.Q.T)
        # > [[-0.16666667  0.          0.5         0.16666667  0.83333333]
        #  [-0.3592106  -0.89802651 -0.1796053   0.1796053   0.        ]
        #  [-0.46201575  0.2746296  -0.70613094 -0.25701446  0.38267831]]
        print(qr.R.T)
        # > [[ 6.          0.          0.        ]
        #  [ 0.         -5.56776436  0.        ]
        #  [ 4.16666667  3.59210604 -6.46031447]]
     */
}
