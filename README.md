# Interplanetary Free Space Optical Network Simulation

## Overview

This simulation models a Free Space Optical (FSO) communication network architecture for Earth-Mars interplanetary communications with advanced controller coordination and traffic management capabilities.

The simulation validates network resilience during solar conjunction events and demonstrates the effectiveness of strategically placed relay satellites at Sun-Earth and Sun-Mars Lagrange points to maintain continuous connectivity.

## Thesis Context

This code implements the network architecture described in the thesis:

**"Enhanced Interplanetary Communication Networks Using Free Space Optical Links with Lagrange Point Relay Stations"**

### Key Contributions Validated by This Simulation

- Solar conjunction resilience through Lagrange point relay positioning
- Controller-coordinated navigation for improved link acquisition
- Dynamic traffic management for optimal bandwidth utilization
- Realistic FSO link budget calculations using Majumdar (2005) equations
- Multi-hop routing through strategic relay infrastructure

## Network Architecture

### Constellation Components

#### 1. Earth Satellites (8 satellites)

- Walker 8/2/1 constellation
- Geostationary orbit (35,786 km altitude)
- 55° inclination for global coverage
- Distributed across 2 orbital planes

#### 2. Mars Satellites (4 satellites)

- Evenly-spaced constellation
- Areostationary orbit (17,034 km altitude)
- 55° inclination
- 90° separation for continuous coverage

#### 3. Earth-Moon Controllers (3 satellites)

- Positioned at Lagrange points L3, L4, L5
- Provide navigation coordination for Earth satellites
- Manage traffic scheduling and handoffs
- Reduce acquisition time by 50% through predictive ephemeris

#### 4. Earth-Sun Relays (4 satellites)

- Positioned at Lagrange points L1, L2, L4, L5
- Bridge Earth system to interplanetary backbone
- 1-meter aperture telescopes with 150W laser power
- Support up to 10 simultaneous FSO links

#### 5. Mars-Sun Relays (4 satellites)

- Positioned at Lagrange points L1, L2, L4, L5
- Bridge Mars system to interplanetary backbone
- Critical for solar conjunction resilience
- Enable continuous Earth-Mars communication

## System Requirements

### Software Requirements

- Python 3.8 or higher
- See `requirements.txt` for package dependencies

### Hardware Requirements

- **RAM:** Minimum 8 GB (16 GB recommended)
- **Disk Space:** ~500 MB for results
- **CPU:** Multi-core recommended for faster simulation

### Key Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Numerical computations |
| scipy | Orbital integration (odeint) |
| astropy | JPL ephemeris data (DE440) |
| networkx | Graph-based connectivity analysis |
| spacepy | Coordinate transformations (optional) |

## Installation

### Step 1: Install Python Packages
```bash
pip install -r requirements.txt
```

### Step 2: SpacePy Installation (if needed)

SpacePy can be challenging to install. Try:
```bash
conda install -c conda-forge spacepy
```

Or see `requirements.txt` for system-specific instructions.

### Step 3: First Run - Ephemeris Download

On first execution, astropy will automatically download the JPL DE440 ephemeris file (~115 MB). This is a one-time download.

## Running the Simulation

### Basic Execution
```bash
python interplanetary_fso_network.py
```

### Output Files

All results are written to the `results/` directory:

#### 1. Time Series Data

- `thesis_data_time_series.csv`: Complete network metrics over time
- `thesis_data_connectivity_comparison.csv`: Earth vs Mars connectivity
- `thesis_data_fso_performance.csv`: FSO link performance metrics

#### 2. Validation Tests

- `validation_summary.csv`: Test results summary
- `validation_orbital_configs.csv`: Configuration impact analysis
- `validation_failure_recovery.csv`: Link recovery metrics
- `validation_traffic_overload.csv`: Traffic stress test results

#### 3. Advanced Metrics

- `thesis_data_link_stability.csv`: Temporal link analysis
- `thesis_data_handoff_performance.csv`: Controller handoff metrics
- `thesis_data_controller_load.csv`: Load balancing statistics
- `thesis_data_topology.csv`: Network topology evolution
- `thesis_data_queue_performance.csv`: Traffic queue analysis
- `thesis_data_occlusion_events.csv`: Solar/planetary blocking events
- `thesis_data_coverage.csv`: Surface coverage metrics
- `thesis_data_controller_assignments.csv`: Satellite-controller assignments

#### 4. Console Output

- `simulation_output_TIMESTAMP.txt`: Complete simulation log

## Key Simulation Parameters

Located at the top of the code file:

### Constellation Size
```python
EARTH_SATS = 8                  # Earth Walker constellation
MARS_SATS = 4                   # Mars constellation
EARTH_MOON_CONTROLLERS = 3      # Navigation controllers
EARTH_SUN_RELAYS = 4            # Earth-Sun Lagrange relays
MARS_SUN_RELAYS = 4             # Mars-Sun Lagrange relays
```

### Orbital Parameters
```python
EARTH_ORBIT_ALTITUDE = 35786.0  # km (GEO)
MARS_ORBIT_ALTITUDE = 17034.0   # km (Areostationary)
```

### FSO Communication
```python
FSO_WAVELENGTH = 1550e-9         # 1550 nm (telecom band)
FSO_BEAM_DIVERGENCE = 30e-6      # 30 microradians
FSO_TRANSMIT_POWER = 100.0       # Watts (satellites)
FSO_APERTURE_DIAMETER = 1.0      # Meters (relays)
FSO_DATA_RATE_BASELINE = 1e9     # 1 Gbps baseline
```

### Simulation Duration
```python
SIM_DURATION = 20000    # Hours (~2.25 years)
SIM_STEP = 0.25         # Hours (15 minutes)
```

## Validation Tests

The simulation includes four critical validation tests (Section 4.2 of thesis):

### Test 1: Solar Conjunction Resilience

- Validates network maintains connectivity when Earth-Mars-Sun align
- Tests L4/L5 relay effectiveness at avoiding solar occlusion
- Measures alternative path availability

### Test 2: Dynamic Link Failure Recovery

- Simulates random link failures
- Measures network recovery time
- Tests redundancy and self-healing capabilities

### Test 3: Traffic Overload Scenario

- Applies 2.5x normal traffic load
- Tests traffic management and prioritization
- Measures network stress tolerance

### Test 4: Orbital Configuration Impact

- Tests conjunction, quadrature, opposition configurations
- Analyzes distance vs connectivity relationship
- Validates interplanetary link performance

## Understanding the Output

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Connectivity Percentage | Percentage of satellites with path to opposite planet | >95% |
| FSO Active Links | Number of established optical links | Higher is better |
| Data Rate (Tbps) | Total network throughput | Scales with links |
| Interplanetary Links | Critical Earth-Mars relay connections | >2 for redundancy |
| Controller Coverage | Percentage of Earth satellites under coordination | >90% |
| Acquisition Success Rate | FSO link establishment success percentage | Navigation +5% |

### Performance Indicators

#### Good Performance

- Earth connectivity: 95-100%
- Mars connectivity: 90-100%
- Interplanetary links: 4-8 active
- Controller coverage: >90%

#### Warning Signs

- Connectivity drops below 90%
- No interplanetary links during conjunction
- Controller coverage <80%
- High link failure rate

## Physics and Models

### Orbital Mechanics

- JPL DE440 ephemeris for planetary positions
- J2 perturbation for Earth oblateness
- Solar radiation pressure modeling
- Third-body gravitational effects
- Relativistic corrections (Schwarzschild)

### FSO Link Budget

Based on Majumdar (2005):

- Transmitter gain: G<sub>T</sub> = 16/θ<sub>T</sub>²
- Receiver gain: G<sub>R</sub> = (πD/λ)²
- Free space loss: S = (λ/4πL)²
- Received power: P<sub>REC</sub> = P<sub>T</sub> × G<sub>T</sub> × τ<sub>T</sub> × τ<sub>ATM</sub> × S × G<sub>R</sub> × τ<sub>R</sub>

### Line-of-Sight Checking

- Ray-sphere intersection for Sun occlusion
- Earth and Mars blocking detection
- Moon occlusion for Earth system links
- Solar exclusion zone: 5× solar radius

## Modifying the Simulation

### Change Constellation Size
```python
EARTH_SATS = 12  # Increase to 12 satellites
MARS_SATS = 6    # Increase to 6 satellites
```

### Adjust FSO Performance
```python
FSO_TRANSMIT_POWER = 200.0      # Increase power to 200W
FSO_APERTURE_DIAMETER = 1.5     # Larger telescope (1.5m)
FSO_BEAM_DIVERGENCE = 20e-6     # Tighter beam (20 μrad)
```

### Run Shorter Test
```python
SIM_DURATION = 1000  # ~1.5 months
SIM_STEP = 1.0       # 1 hour steps
```

### Enable/Disable Components
```python
EARTH_MOON_CONTROLLERS = 0  # Disable controllers
EARTH_SUN_RELAYS = 2        # Reduce to 2 relays
MARS_SUN_RELAYS = 2         # Reduce to 2 relays
```

## Troubleshooting

### Issue: "SpacePy import error"

**Solution:** Install via conda or comment out spacepy-specific coordinate transformations (ticks conversion). The simulation will still run.

### Issue: "Astropy downloading data"

**Solution:** This is normal on first run. JPL ephemeris file is being downloaded. Allow ~5 minutes depending on connection speed.

### Issue: "Low connectivity percentages"

**Solution:** This may be normal during:

- Simulation initialization (first few time steps)
- Solar conjunction configurations
- Link acquisition phases (30-minute delay)

### Issue: "Memory error"

**Solution:** Reduce simulation parameters:
```python
SIM_DURATION = 5000  # Shorter simulation
SIM_STEP = 0.5       # Larger time steps
```

### Issue: "Slow execution"

**Solution:** The simulation is computationally intensive. Expected runtime:

- **Full simulation** (20,000 hours): 2-4 hours
- **Test simulation** (1,000 hours): 10-20 minutes

## Expected Results

For a properly configured network:

### 1. Connectivity

| Planet | Expected Range |
|--------|---------------|
| Earth | 95-100% |
| Mars | 90-100% |
| Overall | >95% |

### 2. Solar Conjunction Test

- **Status:** PASS (maintains connectivity via L4/L5 relays)
- **Alternative paths:** 2-4 routes available
- **Blocked direct links:** Expected

### 3. Link Recovery Test

- **Recovery time:** <2 hours
- **Network self-healing:** SUCCESS
- **Redundancy:** Multiple path restoration

### 4. Traffic Management Test

- **Network stress:** <20% under 2.5x load
- **Priority handling:** EMERGENCY > CRITICAL > HIGH > NORMAL > LOW
- **Queue management:** Effective scheduling

## References

### Key Publications

1. **Majumdar, A. K.** (2005). "Free-space laser communication performance in the atmospheric channel." *Journal of Optical and Fiber Communications Reports*.

2. **Walker, J. G.** (1984). "Satellite constellations." *Journal of the British Interplanetary Society*.

3. **JPL Planetary and Lunar Ephemerides**  
   https://ssd.jpl.nasa.gov/planets/eph_export.html

## Contact and Support

For questions about:

- **Simulation code:** Review comments in the Python file
- **Thesis methodology:** See thesis document Sections 3-4
- **Result interpretation:** See thesis document Section 5

## Version History

### Version 1.0 (Current) - Enhanced FSO with Traffic Management

**Features:**

- Walker constellation for Earth satellites
- Lagrange point relay infrastructure
- Controller-coordinated navigation
- Traffic management and scheduling
- Inter-controller coordination
- Comprehensive validation test suite
- Realistic orbital mechanics with perturbations
- Line-of-sight occlusion modeling

## License and Citation

If using this simulation for academic purposes, please cite the associated thesis:
```
N. D. Zayfman, "A proposed controller-based free-space optical network architecture with 
    Lagrange point relays for Earth-Mars communication," M.S. thesis, Dept. Comput. Sci., 
    Johns Hopkins Univ., Baltimore, MD, USA, 2025.
```
MIT License

Copyright (c) [2025] [Nicholas Zayfman]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

**Note:** This simulation is part of academic research on interplanetary communication networks.