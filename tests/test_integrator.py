import torch
import numpy as np
from torchmd.integrator import kinetic_energy, Integrator
from torchmd.systems import System


def test_kinetic_energy_single_replica():
    """Test kinetic_energy with a single replica"""
    # Create test data: 3 atoms, 1 replica
    masses = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)  # (natoms, 1)
    velocities = torch.tensor(
        [[[1.0, 2.0, 3.0], [0.5, 1.0, 1.5], [2.0, 1.0, 0.5]]], dtype=torch.float32
    )  # (1, natoms, 3)

    # Calculate expected kinetic energy manually
    # E_kin = 0.5 * sum(mass * (vx^2 + vy^2 + vz^2))
    expected = 0.0
    for i in range(3):  # 3 atoms
        mass = masses[i, 0]
        vx, vy, vz = velocities[0, i]
        v_squared = vx * vx + vy * vy + vz * vz
        expected += 0.5 * mass * v_squared

    result = kinetic_energy(masses, velocities)
    assert result.shape == (1, 1)
    np.testing.assert_allclose(result.numpy(), [[expected]], rtol=1e-6)


def test_kinetic_energy_multiple_replicas():
    """Test kinetic_energy with multiple replicas"""
    # Create test data: 2 atoms, 2 replicas
    masses = torch.tensor([[1.0], [2.0]], dtype=torch.float32)  # (natoms, 1)
    velocities = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
        dtype=torch.float32,
    )  # (2, natoms, 3)

    # Calculate expected kinetic energies manually for each replica
    expected = []
    for replica in range(2):
        e_kin = 0.0
        for atom in range(2):
            mass = masses[atom, 0]
            vx, vy, vz = velocities[replica, atom]
            v_squared = vx * vx + vy * vy + vz * vz
            e_kin += 0.5 * mass * v_squared
        expected.append(e_kin)

    result = kinetic_energy(masses, velocities)
    assert result.shape == (2, 1)
    np.testing.assert_allclose(
        result.numpy(), [[expected[0]], [expected[1]]], rtol=1e-6
    )


def test_kinetic_energy_zero_velocity():
    """Test kinetic_energy with zero velocities"""
    masses = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    velocities = torch.zeros(1, 2, 3, dtype=torch.float32)

    result = kinetic_energy(masses, velocities)
    assert result.shape == (1, 1)
    np.testing.assert_allclose(result.numpy(), [[0.0]], rtol=1e-6)


def test_kinetic_energy_with_batch():
    """Test kinetic_energy with batch grouping atoms within replicas"""
    masses = torch.tensor([[1.0], [2.0], [1.5]], dtype=torch.float32)  # (natoms, 1)
    velocities = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],  # replica 0
            [[0.5, 0.0, 0.0], [0.0, 1.5, 0.0], [1.0, 0.0, 0.0]],  # replica 1
        ],
        dtype=torch.float32,
    )  # (2, natoms, 3)

    # Group atoms within each replica into batches (e.g., different atom types)
    batch = torch.tensor(
        [0, 0, 1], dtype=torch.long
    )  # atom 0,1 -> batch 0, atom 2 -> batch 1

    result = kinetic_energy(masses, velocities, batch=batch)
    assert result.shape == (2, 2)  # (nreplicas, nbatches)

    # Calculate expected values
    expected = torch.zeros(2, 2, dtype=torch.float32)

    for replica in range(2):  # 2 replicas
        for batch_idx in range(2):  # 2 batches
            ke_sum = 0.0
            for atom_idx in range(3):  # 3 atoms
                if batch[atom_idx] == batch_idx:  # atom belongs to this batch
                    mass = masses[atom_idx, 0]
                    vx, vy, vz = velocities[replica, atom_idx]
                    v_squared = vx * vx + vy * vy + vz * vz
                    ke_sum += 0.5 * mass * v_squared
            expected[replica, batch_idx] = ke_sum

    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6)


def test_kinetic_energy_empty_batches():
    """Test batched_kinetic_energy with some empty batches"""
    masses = torch.tensor([[1.0], [2.0]], dtype=torch.float32)  # (natoms, 1)
    velocities = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32
    )  # (1, natoms, 3)
    batch = torch.tensor(
        [0, 2], dtype=torch.long
    )  # atom 0 -> batch 0, atom 1 -> batch 2

    result = kinetic_energy(masses, velocities, batch=batch)
    assert result.shape == (1, 3)  # (nreplicas, nbatches)

    # Batch 1 should be 0 for replica 0
    assert result[0, 1].item() == 0.0


def test_kinetic_energy_single_atom_batches():
    """Test batched_kinetic_energy where each atom is in its own batch"""
    masses = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)  # (natoms, 1)
    velocities = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]], dtype=torch.float32
    )  # (1, natoms, 3)
    batch = torch.tensor([0, 1, 2], dtype=torch.long)

    result = kinetic_energy(masses, velocities, batch=batch)
    assert result.shape == (1, 3)  # (nreplicas, nbatches)

    # Calculate expected values using the actual arrays
    # Each atom is in its own batch, so each batch contains KE from one atom
    expected = torch.zeros(1, 3, dtype=torch.float32)  # (nreplicas, nbatches)

    for batch_idx in range(3):  # 3 batches (one per atom)
        atom_idx = batch_idx  # since batch[i] = i for this test
        mass = masses[atom_idx, 0]
        vx, vy, vz = velocities[0, atom_idx]
        v_squared = vx * vx + vy * vy + vz * vz
        expected[0, batch_idx] = 0.5 * mass * v_squared
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)


def test_integrator_initialization():
    """Test Integrator class initialization"""
    # Create a simple system
    natoms = 2
    nreplicas = 1
    device = "cpu"
    precision = torch.float32

    system = System(natoms, nreplicas, precision, device)
    system.set_masses(torch.tensor([1.0, 2.0]))

    # Create a mock forces object
    class MockForces:
        def compute(self, pos, box, forces):
            # Return zero potential energy for simplicity
            return 0.0

    forces = MockForces()
    timestep = 0.001  # ps

    # Test initialization without temperature control
    integrator = Integrator(system, forces, timestep, device)
    assert integrator.systems is system
    assert integrator.forces is forces
    assert integrator.device == device
    assert integrator.T is None
    assert integrator.gamma is None

    # Test initialization with temperature control
    T = 300.0
    gamma = 1.0
    integrator_with_temp = Integrator(
        system, forces, timestep, device, gamma=gamma, T=T
    )
    assert integrator_with_temp.T == T
    assert integrator_with_temp.gamma is not None


def test_integrator_step():
    """Test Integrator step method"""
    # Create a simple system with known initial conditions
    natoms = 2
    nreplicas = 1
    device = "cpu"
    precision = torch.float32

    system = System(natoms, nreplicas, precision, device)

    # Set some initial velocities
    velocities = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    system.set_velocities(velocities)
    system.set_masses(torch.tensor([1.0, 2.0]))

    # Create a mock forces object
    class MockForces:
        def compute(self, pos, box, forces):
            # Set some non-zero forces to test integration
            forces.copy_(
                torch.tensor(
                    [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=precision, device=device
                )
            )
            return 1.5  # Return some potential energy

    forces = MockForces()
    timestep = 0.001  # ps

    integrator = Integrator(system, forces, timestep, device)

    # Test single step
    Ekin, pot, T = integrator.step(niter=1)

    # Check return types and shapes
    assert isinstance(Ekin, np.ndarray)
    assert isinstance(pot, (int, float))
    assert isinstance(T, np.ndarray)
    assert Ekin.shape == (nreplicas,)
    assert T.shape == (nreplicas,)

    # Check that velocities changed (integration occurred)
    assert not torch.allclose(system.vel, velocities)

    # Check kinetic energy calculation
    expected_kinetic = kinetic_energy(system.masses, system.vel)
    np.testing.assert_allclose(Ekin, expected_kinetic.numpy().flatten(), rtol=1e-6)

    # Test multiple steps
    initial_pos = system.pos.clone()
    Ekin_multi, pot_multi, T_multi = integrator.step(niter=3)

    # Positions should have changed after multiple steps
    assert not torch.allclose(system.pos, initial_pos)

    # Check that multiple steps return final state
    assert Ekin_multi.shape == (nreplicas,)
    assert T_multi.shape == (nreplicas,)


def test_integrator_with_batches():
    """Test Integrator class with batch parameter"""
    # Create a system with 3 atoms that can be grouped into batches
    natoms = 3
    nreplicas = 1
    device = "cpu"
    precision = torch.float32

    system = System(natoms, nreplicas, precision, device)

    # Set initial velocities
    velocities = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]])
    system.set_velocities(velocities)
    system.set_masses(torch.tensor([1.0, 2.0, 1.5]))

    # Create batch indices: atoms 0,1 in batch 0, atom 2 in batch 1
    batch = torch.tensor([0, 0, 1], dtype=torch.long, device=device)

    # Create a mock forces object
    class MockForces:
        def compute(self, pos, box, forces):
            # Set significant forces to test integration (much larger to see velocity change)
            forces.copy_(
                torch.tensor(
                    [[[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]]],
                    dtype=precision,
                    device=device,
                )
            )
            return 2.5  # Return some potential energy

    forces = MockForces()
    timestep = 0.001  # ps

    # Create integrator with batch parameter
    integrator = Integrator(system, forces, timestep, device, batch=batch)

    # Verify batch setup
    assert integrator.batch is batch
    assert np.array_equal(
        integrator.natoms, np.array([2, 1])
    )  # 2 atoms in batch 0, 1 atom in batch 1

    # Test single step
    Ekin, pot, T = integrator.step(niter=1)

    # Check return types and shapes
    assert isinstance(Ekin, np.ndarray)
    assert isinstance(pot, (int, float))
    assert isinstance(T, np.ndarray)
    assert Ekin.shape == (2,)  # (nbatches,) - kinetic energy per batch
    assert T.shape == (2,)  # (nbatches,) - temperature per batch

    # Check that velocities changed (integration occurred)
    assert not torch.allclose(system.vel, velocities)

    # Check kinetic energy calculation with batching
    expected_kinetic = kinetic_energy(system.masses, system.vel, batch=batch)
    np.testing.assert_allclose(Ekin, expected_kinetic.numpy().flatten(), rtol=1e-6)

    # Verify the kinetic energy values make sense
    # Batch 0: atoms 0,1; Batch 1: atom 2
    assert Ekin[0] > 0  # Should have significant KE from atoms 0,1
    assert Ekin[1] > 0  # Should have KE from atom 2

    # Test that potential energy is returned correctly
    assert pot == 2.5


def test_integrator_deterministic_two_atoms():
    """Test Integrator with two atoms, no temperature, deterministic forces"""
    # Create system with 2 atoms, 1 replica
    natoms = 2
    nreplicas = 1
    device = "cpu"
    precision = torch.float32

    system = System(natoms, nreplicas, precision, device)

    # Set initial positions: atom 0 at (0,0,0), atom 1 at (1,1,1)
    initial_pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=precision)
    system.set_positions(initial_pos[:, :, None])

    # Set initial velocities: both atoms moving with velocity (0.1, 0.2, 0.3)
    initial_vel = torch.tensor([[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]], dtype=precision)
    system.set_velocities(initial_vel)

    # Set masses: atom 0 has mass 1.0, atom 1 has mass 2.0
    masses = torch.tensor([1.0, 2.0], dtype=precision)
    system.set_masses(masses)

    # Define constant forces that don't depend on position
    # Force on atom 0: (1.0, 2.0, 3.0), atom 1: (-0.5, 1.5, -2.5)
    constant_forces = torch.tensor(
        [[[1.0, 2.0, 3.0], [-0.5, 1.5, -2.5]]], dtype=precision, device=device
    )

    # Create mock forces object that returns constant forces
    class ConstantForces:
        def __init__(self, forces):
            self.constant_forces = forces

        def compute(self, pos, box, forces_tensor):
            # Always return the same forces regardless of position
            forces_tensor.copy_(self.constant_forces)
            return 0.0  # potential energy doesn't matter for this test

    forces = ConstantForces(constant_forces)
    timestep = 0.002  # ps (small timestep for accuracy)

    # Initialize forces at the starting positions
    forces.compute(system.pos, system.box, system.forces)

    # Create integrator without temperature control
    integrator = Integrator(system, forces, timestep, device, T=None)

    # Store initial state
    pos_before = system.pos.clone()
    vel_before = system.vel.clone()

    # Perform one integration step
    Ekin, pot, T = integrator.step(niter=1)

    # Calculate expected positions and velocities manually
    dt = timestep / 48.88821  # Convert to internal time units

    # Calculate accelerations
    accel = constant_forces / system.masses.unsqueeze(0)  # (1, 2, 3)

    # Velocity Verlet algorithm:
    # 1. First half-step: update positions and velocities
    # pos += vel * dt + 0.5 * accel * dt * dt
    # vel += 0.5 * dt * accel
    expected_pos = pos_before + vel_before * dt + 0.5 * accel * dt * dt

    # 2. Forces are recomputed (but stay the same for constant forces)
    # 3. Second half-step: update velocities
    # vel += 0.5 * dt * accel
    expected_vel = vel_before + accel * dt

    # Check that final positions match expected
    np.testing.assert_allclose(
        system.pos.numpy(),
        expected_pos.numpy(),
        rtol=1e-6,
        err_msg="Final positions do not match expected values",
    )

    # Check that final velocities match expected
    np.testing.assert_allclose(
        system.vel.numpy(),
        expected_vel.numpy(),
        rtol=1e-6,
        err_msg="Final velocities do not match expected values",
    )

    # Verify that kinetic energy is calculated correctly
    expected_kinetic = kinetic_energy(system.masses, system.vel)
    np.testing.assert_allclose(Ekin, expected_kinetic.numpy().flatten(), rtol=1e-6)

    # Verify temperature is calculated correctly
    from torchmd.integrator import kinetic_to_temp

    expected_temp = kinetic_to_temp(Ekin, natoms)
    np.testing.assert_allclose(T, expected_temp, rtol=1e-6)


def test_integrator_deterministic_two_atoms_two_replicas():
    """Test Integrator with two atoms, two replicas, no temperature, deterministic forces"""
    # Create system with 2 atoms, 2 replicas
    natoms = 2
    nreplicas = 2
    device = "cpu"
    precision = torch.float32

    system = System(natoms, nreplicas, precision, device)

    # Set initial positions: same for both replicas
    # atom 0 at (0,0,0), atom 1 at (1,1,1)
    initial_pos = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=precision)
    system.pos.copy_(initial_pos.repeat(nreplicas, 1, 1))

    # Set initial velocities: same for both replicas
    # both atoms moving with velocity (0.1, 0.2, 0.3)
    initial_vel = torch.tensor([[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]], dtype=precision)
    system.vel.copy_(initial_vel.repeat(nreplicas, 1, 1))

    # Set masses: atom 0 has mass 1.0, atom 1 has mass 2.0 (same for both replicas)
    masses = torch.tensor([1.0, 2.0], dtype=precision)
    system.set_masses(masses)

    # Define different constant forces for each replica
    # Replica 0: Force on atom 0: (1.0, 2.0, 3.0), atom 1: (-0.5, 1.5, -2.5)
    # Replica 1: Force on atom 0: (0.5, 1.0, 1.5), atom 1: (-1.0, 2.0, -3.0)
    constant_forces = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [-0.5, 1.5, -2.5]],  # replica 0
            [[0.5, 1.0, 1.5], [-1.0, 2.0, -3.0]],
        ],  # replica 1
        dtype=precision,
        device=device,
    )

    # Create mock forces object that returns constant forces
    class ConstantForces:
        def __init__(self, forces):
            self.constant_forces = forces

        def compute(self, pos, box, forces_tensor):
            # Always return the same forces regardless of position
            forces_tensor.copy_(self.constant_forces)
            return 0.0  # potential energy doesn't matter for this test

    forces = ConstantForces(constant_forces)
    timestep = 0.002  # ps (small timestep for accuracy)

    # Initialize forces at the starting positions
    forces.compute(system.pos, system.box, system.forces)

    # Create integrator without temperature control
    integrator = Integrator(system, forces, timestep, device, T=None)

    # Store initial state
    pos_before = system.pos.clone()
    vel_before = system.vel.clone()

    # Perform one integration step
    Ekin, pot, T = integrator.step(niter=1)

    # Calculate expected positions and velocities manually for each replica
    dt = timestep / 48.88821  # Convert to internal time units

    # Calculate accelerations for each replica
    accel = constant_forces / system.masses.unsqueeze(0)  # (2, 2, 3)

    # Velocity Verlet algorithm:
    # 1. First half-step: update positions and velocities
    # pos += vel * dt + 0.5 * accel * dt * dt
    # vel += 0.5 * dt * accel
    expected_pos = pos_before + vel_before * dt + 0.5 * accel * dt * dt

    # 2. Forces are recomputed (but stay the same for constant forces)
    # 3. Second half-step: update velocities
    # vel += 0.5 * dt * accel
    expected_vel = vel_before + accel * dt

    # Check that final positions match expected for both replicas
    np.testing.assert_allclose(
        system.pos.numpy(),
        expected_pos.numpy(),
        rtol=1e-6,
        err_msg="Final positions do not match expected values",
    )

    # Check that final velocities match expected for both replicas
    np.testing.assert_allclose(
        system.vel.numpy(),
        expected_vel.numpy(),
        rtol=1e-6,
        err_msg="Final velocities do not match expected values",
    )

    # Verify that kinetic energy is calculated correctly for each replica
    expected_kinetic = kinetic_energy(system.masses, system.vel)
    np.testing.assert_allclose(Ekin, expected_kinetic.numpy().flatten(), rtol=1e-6)

    # Verify temperature is calculated correctly for each replica
    from torchmd.integrator import kinetic_to_temp

    expected_temp = kinetic_to_temp(Ekin, natoms)
    np.testing.assert_allclose(T, expected_temp, rtol=1e-6)
