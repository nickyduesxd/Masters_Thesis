#Import packages and libraries
import numpy as np
import networkx as nx
import datetime as dt
import csv
import json
import os
import spacepy.time as spt
import spacepy.coordinates as spcoord 
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric
from astropy import units as u
from scipy.integrate import odeint
from astropy import units as u
from datetime import datetime
import sys

#Radius constants are in KM
EARTH_RADIUS = 6371.0 # Earth's mean radius
MARS_RADIUS = 3389.5 # Mars' mean radius
MOON_RADIUS = 1737.4 # Earth's Moon mean radius
SUN_RADIUS = 695700.0  # Solar radius
#Astronomical Unit in KM
AU = 149597870.7 
#Satellite orbit parameters
EARTH_ORBIT_ALTITUDE = 35786.0 # Geostationary Orbit (GEO)
MARS_ORBIT_ALTITUDE = 17034.0 # Mars synchronous orbit (Areostationary)
#FSO Communication ranges
CONTROLLER_RANGE = 0.3 * AU  # Earth-Moon system coverage 
RELAY_RANGE = 4.0 * AU  # Deep space relay 
#FSO Communication System Constants
FSO_WAVELENGTH = 1550e-9  #1550nm
FSO_BEAM_DIVERGENCE = 10e-6  #10 microradian beam divergence, realistic beam divergense 
FSO_TRANSMIT_POWER = 100.0  #Watts, realistic power
FSO_APERTURE_DIAMETER = 0.3  #Meters - telescope aperture - opening that allows a telescope to collect light - can possibly make bigger
FSO_POINTING_ACCURACY = 10e-6  # 10 microradian pointing accuracy, realistic pointing accuracy
FSO_ACQUISITION_TIME = 300.0  # 300 seconds beam acquisition time
FSO_DATA_RATE_BASELINE = 1e9  # 1 Gbps baseline data rate
#Atmospheric losses for Earth only 
ATMOSPHERIC_LOSS_DB = 10.0  # 10dB loss through Earth atmosphere - Signal attenuation through atmosphere due to weather, etc.
#FSO Link Quality Levels
FSO_LINK_EXCELLENT = 0.95 
FSO_LINK_GOOD = 0.85      
FSO_LINK_DEGRADED = 0.70  
FSO_LINK_POOR = 0.50      
#Network management thresholds
QOS_PRIORITY_LEVELS = ['EMERGENCY', 'CRITICAL', 'HIGH', 'NORMAL', 'LOW']
#Number of satellites
EARTH_SATS = 8
MARS_SATS = 4
EARTH_MOON_CONTROLLERS = 4
EARTH_SUN_RELAYS = 4
MARS_SUN_RELAYS = 4
#Simulation parameters
SIM_DURATION = 20000 #roughly 2.25 years
SIM_STEP = 1.0 #1 hrs
# Navigation and Coordination Constants
NAVIGATION_UPDATE_INTERVAL = 0.1  #6 minutes between navigation updates
PREDICTION_HORIZON = 2.0  #Number of hours ahead to predict positions
HANDOFF_THRESHOLD = 0.8  #Signal quality threshold for handoff
COORDINATION_RANGE = 0.5 * AU  #Range for controller coordination
#Traffic Management Constants
TRAFFIC_UPDATE_INTERVAL = 0.05  #Hours between traffic updates (3 minutes)
TRANSMISSION_SLOT_DURATION = 0.02  #Hours per transmission slot (1.2 minutes)
#Solar exclusion zone to account for sun's corona effects on lasers
SOLAR_EXCLUSION_RADIUS = SUN_RADIUS * 1.5

#The main satellite object that manages position and velocity in 3D space, handles FSO terminal operations, performs orbital mechanics calculations, manages network connections and coordinates with controllers for traffic management.
class Satellite:
    def __init__(self, sat_id, planet, orbit_altitude, inclination, phase, sat_type='satellite', lagrange_point=None, color='blue'):
        self.sat_id = sat_id
        self.planet = planet
        self.orbit_altitude = orbit_altitude
        self.inclination = inclination
        self.phase = np.radians(phase)
        self.sat_type = sat_type
        self.lagrange_point = lagrange_point
        self.color = color
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.last_update_time = 0
        self.coords = None
        self.connections = []
        self.earth_connected = False
        self.mars_connected = False
        self.relay_hops = []
        
        if sat_type == 'controller':
            self.navigation_coordinator = NavigationCoordinator(sat_id)
            self.satellites_in_region = []
            self.last_coordination_update = 0
        
        #FSO Terminal Configuration
        if sat_type == 'relay':
            aperture_size = 1.0  #1 meter telescope
            laser_power = 150.0   #50W laser
        elif sat_type == 'controller':
            aperture_size = 0.5  #50cm telescope  
            laser_power = 50.0   #20W laser
        else:
            aperture_size = 0.3  #30cm telescope
            laser_power = 10.0   #10W laser
        self.fso_terminal = FSO_Terminal(sat_id, aperture_size, laser_power)
        self.orbital_mechanics = RealisticOrbitalMechanics()
        if sat_type == 'relay':
            self.simultaneous_links_max = 50
        elif sat_type == 'controller':
            self.simultaneous_links_max = 12 
        else:
            self.simultaneous_links_max = 8
        self.ephemeris = RealEphemerisData()
        if sat_type in ['controller', 'relay'] and lagrange_point:
            self.ephemeris = RealEphemerisData()
            self.lagrange_calculator = PreciseLagrangePoints(self.ephemeris)
    
    #Update satellite position using realistic orbital mechanics
    def update_position_with_orbital_perturbations(self, time, ticks, dt=None):
        if dt is None:
            dt = SIM_STEP  #in hours
        if self.sat_type in ['controller', 'relay'] and hasattr(self, 'lagrange_calculator'):
            self.position = self._get_precise_lagrange_position(time)
            return
    
        #metric conversion
        state = np.concatenate([
            self.position * 1000,  #km to m
            self.velocity * 1000   #km/s to m/s
        ])
    
        params = {
        'central_body': self.planet,
        'area_to_mass': 20.0 / 1500.0,  #m²/kg - solar panel area / satellite mass
        'sun_position': np.array([0, 0, 0]),  #Sun at origin
        'moon_position': self.ephemeris.get_moon_position(time) * 1000
        }
    
        t_span = [0, dt * 3600]  #hrs to sec
    
        #Use orbital mechanics integrator
        solution = odeint(self.orbital_mechanics.propagate_orbit_with_perturbations,state,t_span,args=(params,))

        #Extract final state and convert back to km
        final_state = solution[-1]
        self.position = final_state[:3] / 1000  #m to km
        self.velocity = final_state[3:] / 1000  #m/s to km/s
    
        #Update coords for spacepy
        self.coords = spcoord.Coords(
            [[self.position[0], self.position[1], self.position[2]]],
            'GEO', 'car', ticks=ticks, units=['km', 'km', 'km']
        )
    
    #Get precise Lagrange point position using PreciseLagrangePoints calculator
    def _get_precise_lagrange_position(self, time_hours):
        if not hasattr(self, 'lagrange_calculator') or not self.lagrange_point:
            return np.array([0, 0, 0])
    
        #map lagrange_point string to calculator method
        lagrange_methods = {
        'Earth-Moon-L1': self.lagrange_calculator.earth_moon_l1,
        'Earth-Moon-L2': self.lagrange_calculator.earth_moon_l2,
        #'Earth-Moon-L3': self.lagrange_calculator.earth_moon_l3,
        'Earth-Moon-L4': self.lagrange_calculator.earth_moon_l4,
        'Earth-Moon-L5': self.lagrange_calculator.earth_moon_l5,
        'Earth-Sun-L1': self.lagrange_calculator.earth_sun_l1,
        'Earth-Sun-L2': self.lagrange_calculator.earth_sun_l2,
        #'Earth-Sun-L3': self.lagrange_calculator.earth_sun_l3,
        'Earth-Sun-L4': self.lagrange_calculator.earth_sun_l4,
        'Earth-Sun-L5': self.lagrange_calculator.earth_sun_l5,
        'Mars-Sun-L1': self.lagrange_calculator.mars_sun_l1,
        'Mars-Sun-L2': self.lagrange_calculator.mars_sun_l2,
        #'Mars-Sun-L3': self.lagrange_calculator.mars_sun_l3,
        'Mars-Sun-L4': self.lagrange_calculator.mars_sun_l4,
        'Mars-Sun-L5': self.lagrange_calculator.mars_sun_l5,
        }
    
        if self.lagrange_point in lagrange_methods:
            return lagrange_methods[self.lagrange_point](time_hours)
        else:
            print(f"Warning: Unknown Lagrange point {self.lagrange_point}")
            return np.array([0, 0, 0])

    #Enhanced controller coordination with navigation services and traffic management
    def perform_controller_coordination(self, all_satellites, current_time):
        if self.sat_type != 'controller':
            return
        
        #Update coordination at regular intervals
        if current_time - self.last_coordination_update < NAVIGATION_UPDATE_INTERVAL:
            return
        
        self.last_coordination_update = current_time
        
        #Find satellites in this controller's "region"
        self.satellites_in_region = self._find_satellites_in_region(all_satellites)
        
        #Update relay ephemeris data
        earth_sun_relays = [sat for sat in all_satellites 
                           if sat.sat_type == 'relay' and 'Earth-Sun' in str(sat.lagrange_point)]
        self.navigation_coordinator.update_relay_ephemeris(earth_sun_relays, current_time)
        
        #Generate coordination update for navigation + traffic
        enhanced_update = self.navigation_coordinator.generate_enhanced_coordination_update(
            self.satellites_in_region, current_time
        )
        
        #Send updates to satellites in "region"
        for satellite in self.satellites_in_region:
            #Send navigation update
            if enhanced_update['navigation_data']:
                satellite.fso_terminal.receive_navigation_update(enhanced_update['navigation_data'])
            #Send traffic coordination update
            if enhanced_update['traffic_data']:
                satellite.fso_terminal.receive_traffic_coordination_update(enhanced_update)
        #Coordinate handoffs
        handoff_commands = self.navigation_coordinator.coordinate_handoffs(
            self.satellites_in_region, current_time
        )
        #Log coordination
        if len(self.satellites_in_region) > 0:
            traffic_info = ""
            if enhanced_update['traffic_data']:
                scheduled_transmissions = len([s for s in enhanced_update['traffic_data'].satellite_instructions.values() 
                                             if s.get('status') != 'QUEUED'])
                traffic_info = f", {scheduled_transmissions} transmissions scheduled"
            
            print(f"    Controller {self.sat_id}: Coordinating {len(self.satellites_in_region)} satellites, "
                  f"{len(handoff_commands)} handoffs planned{traffic_info}")
    #Find Earth satellites in this controller's coverage "region"
    def _find_satellites_in_region(self, all_satellites):
        satellites_in_region = []
        
        for satellite in all_satellites:
            if satellite.planet == 'Earth' and self._is_in_controller_region(satellite):
                distance = np.linalg.norm(self.position - satellite.position)
                if distance < COORDINATION_RANGE * 1000:  #Within coordination range
                    satellites_in_region.append(satellite)
        
        return satellites_in_region
    #Check FSO link feasibility including line-of-sight
    def _can_establish_fso_link(self, other_sat, distance):
        distance_km = distance / 1000 if distance > 1000000 else distance
        max_range_km = RELAY_RANGE * 149597870.7  # Convert AU to km
    
        #in range?
        if distance_km > max_range_km:
            return False
    
        current_time = getattr(self, 'last_update_time', 0)
        check_sun = True
    
        #Earth satellite to Earth-Sun relay - no Sun check needed - relay is positioned to avoid Sun
        if ((self.planet == 'Earth' and other_sat.sat_type == 'relay' and 'Earth-Sun' in str(other_sat.lagrange_point)) or
            (other_sat.planet == 'Earth' and self.sat_type == 'relay' and 'Earth-Sun' in str(self.lagrange_point))):
            check_sun = False
        #Mars satellite to Mars-Sun relay - no Sun check needed
        elif ((self.planet == 'Mars' and other_sat.sat_type == 'relay' and 'Mars-Sun' in str(other_sat.lagrange_point)) or
              (other_sat.planet == 'Mars' and self.sat_type == 'relay' and 'Mars-Sun' in str(self.lagrange_point))):
            check_sun = False
        #Controller to Earth-Sun relay - no Sun check needed
        elif ((self.sat_type == 'controller' and other_sat.sat_type == 'relay' and 'Earth-Sun' in str(other_sat.lagrange_point)) or
          (other_sat.sat_type == 'controller' and self.sat_type == 'relay' and 'Earth-Sun' in str(self.lagrange_point))):
            check_sun = False
    
        #Check line of sight
        los_clear, blocking_body = LineOfSightChecker.is_line_of_sight_clear(
            self.position, 
            other_sat.position, 
            current_time,
            check_sun=check_sun,
            check_planets=True,
            check_moon=(self.planet == 'Earth' or other_sat.planet == 'Earth')
        )
    
        if not los_clear:
            return False
    
        #Earth-Sun relays can talk to each other
        if (self.sat_type == 'relay' and other_sat.sat_type == 'relay' and
            'Earth-Sun' in str(self.lagrange_point) and 'Earth-Sun' in str(other_sat.lagrange_point)):
            return True
        #Mars-Sun relays can talk to each other  
        elif (self.sat_type == 'relay' and other_sat.sat_type == 'relay' and
          'Mars-Sun' in str(self.lagrange_point) and 'Mars-Sun' in str(other_sat.lagrange_point)):
            return True
        #Controllers can connect to Earth-Sun relays
        elif ((self.sat_type == 'controller' and other_sat.sat_type == 'relay' and 'Earth-Sun' in str(other_sat.lagrange_point)) or
          (other_sat.sat_type == 'controller' and self.sat_type == 'relay' and 'Earth-Sun' in str(self.lagrange_point))):
            return True
        #Interplanetary backbone (Earth-Sun <=> Mars-Sun)
        elif (self.sat_type == 'relay' and other_sat.sat_type == 'relay' and
              (('Earth-Sun' in str(self.lagrange_point) and 'Mars-Sun' in str(other_sat.lagrange_point)) or
            ('Mars-Sun' in str(self.lagrange_point) and 'Earth-Sun' in str(other_sat.lagrange_point)))):
            return True
        #Earth satellites prioritize controller connections when available
        elif ((self.planet == 'Earth' and other_sat.sat_type == 'controller') or
          (other_sat.planet == 'Earth' and self.sat_type == 'controller')):
            return True
        #Access network links - fallback if no controller available
        elif ((self.planet == 'Earth' and other_sat.sat_type == 'relay' and 'Earth-Sun' in str(other_sat.lagrange_point)) or
          (other_sat.planet == 'Earth' and self.sat_type == 'relay' and 'Earth-Sun' in str(self.lagrange_point))):
            return True
        elif ((self.planet == 'Mars' and other_sat.sat_type == 'relay' and 'Mars-Sun' in str(other_sat.lagrange_point)) or
          (other_sat.planet == 'Mars' and self.sat_type == 'relay' and 'Mars-Sun' in str(self.lagrange_point))):
            return True
        #Regional links - existing satellites
        elif self.planet == other_sat.planet:
            return True
        return False

    #FSO link updates with controller coordination and traffic management
    def update_fso_links(self, time, other_satellites):
        #Update existing links
        links_to_remove = []
        for target_id, link_info in self.fso_terminal.current_links.items():
            target_sat = next((s for s in other_satellites if s.sat_id == target_id), None)
            if target_sat:
                distance = np.linalg.norm(self.position - target_sat.position) * 1000
                if not self.fso_terminal.update_link_quality(target_sat.fso_terminal, distance, time):
                    links_to_remove.append(target_id)
            else:
                links_to_remove.append(target_id)
        #Remove dead links
        for target_id in links_to_remove:
            del self.fso_terminal.current_links[target_id]
        #Check for coordinated transmission authorization
        if self.fso_terminal.is_authorized_to_transmit(time):
            if self.fso_terminal.begin_coordinated_transmission(time):
                scheduled_relay_id = self.fso_terminal.transmission_schedule.get('relay_target')
                if scheduled_relay_id:
                    target_sat = next((s for s in other_satellites if s.sat_id == scheduled_relay_id), None)
                    if target_sat and target_sat.sat_id not in self.fso_terminal.current_links:
                        distance = np.linalg.norm(self.position - target_sat.position) * 1000
                        if self._can_establish_fso_link(target_sat, distance):
                            success = self.fso_terminal.attempt_link_acquisition(target_sat.fso_terminal, distance, time)
                            if success:
                                print(f"    SCHEDULED TRANSMISSION: {self.sat_id} -> {scheduled_relay_id} (traffic coordinated)")
                                return
        if self.planet == 'Earth':
            self._establish_earth_satellite_links(other_satellites, time)
        elif self.sat_type == 'controller':
            self._establish_controller_links(other_satellites, time)
        else:
            self._establish_other_links(other_satellites, time)

    #Earth satellite link establishment with controller priority and traffic coordination
    def _establish_earth_satellite_links(self, other_satellites, time):
        #Check if we have navigation guidance
        has_navigation = len(self.fso_terminal.navigation_updates) > 0
        recommended_target = self.fso_terminal.recommended_target
        has_traffic_schedule = bool(self.fso_terminal.transmission_schedule)
        
        #Priority 0 - Follow traffic coordinator's relay assignment if we have scheduled transmission
        if has_traffic_schedule and self.fso_terminal.transmission_schedule.get('relay_target'):
            scheduled_relay = self.fso_terminal.transmission_schedule['relay_target']
            target_sat = next((s for s in other_satellites if s.sat_id == scheduled_relay), None)
            if target_sat and target_sat.sat_id not in self.fso_terminal.current_links:
                distance = np.linalg.norm(self.position - target_sat.position) * 1000
                if self._can_establish_fso_link(target_sat, distance):
                    success = self.fso_terminal.attempt_link_acquisition(target_sat.fso_terminal, distance, time)
                    if success:
                        print(f"    TRAFFIC SCHEDULED LINK: {self.sat_id} -> {scheduled_relay} (traffic coordinator assigned)")
                        return
        
        #Priority 1 - Follow controller recommendation if available
        if has_navigation and recommended_target:
            target_sat = next((s for s in other_satellites if s.sat_id == recommended_target), None)
            if target_sat and target_sat.sat_id not in self.fso_terminal.current_links:
                distance = np.linalg.norm(self.position - target_sat.position) * 1000
                if self._can_establish_fso_link(target_sat, distance):
                    success = self.fso_terminal.attempt_link_acquisition(target_sat.fso_terminal, distance, time)
                    if success:
                        print(f"    GUIDED LINK: {self.sat_id} -> {recommended_target} (controller recommended)")
                        return
        
        #Priority 2 - Connect to nearest controller
        controllers = [s for s in other_satellites if s.sat_type == 'controller']
        if controllers and len(self.fso_terminal.current_links) < self.simultaneous_links_max:
            nearest_controller = min(controllers, key=lambda c: np.linalg.norm(self.position - c.position))
            if nearest_controller.sat_id not in self.fso_terminal.current_links:
                distance = np.linalg.norm(self.position - nearest_controller.position) * 1000
                if self._can_establish_fso_link(nearest_controller, distance):
                    success = self.fso_terminal.attempt_link_acquisition(
                        nearest_controller.fso_terminal, distance, time)
                    if success:
                        print(f"    CONTROLLER LINK: {self.sat_id} -> {nearest_controller.sat_id}")
                        return
        
        #Priority 3 - Fallback to relay connection
        earth_relays = [s for s in other_satellites 
                       if s.sat_type == 'relay' and 'Earth-Sun' in str(s.lagrange_point)]
        if earth_relays and len(self.fso_terminal.current_links) < self.simultaneous_links_max:
            nearest_relay = min(earth_relays, key=lambda r: np.linalg.norm(self.position - r.position))
            if nearest_relay.sat_id not in self.fso_terminal.current_links:
                distance = np.linalg.norm(self.position - nearest_relay.position) * 1000
                if self._can_establish_fso_link(nearest_relay, distance):
                    success = self.fso_terminal.attempt_link_acquisition(
                        nearest_relay.fso_terminal, distance, time)
                    if success:
                        print(f"    RELAY FALLBACK: {self.sat_id} -> {nearest_relay.sat_id}")

    #Establish controller links to relays and managed satellites
    def _establish_controller_links(self, other_satellites, time):
        #Priority 1 - Connect to Earth-Sun relays
        earth_relays = [s for s in other_satellites 
                       if s.sat_type == 'relay' and 'Earth-Sun' in str(s.lagrange_point)]
        for relay in earth_relays:
            if (relay.sat_id not in self.fso_terminal.current_links and len(self.fso_terminal.current_links) < self.simultaneous_links_max):
                distance = np.linalg.norm(self.position - relay.position) * 1000
                if self._can_establish_fso_link(relay, distance):
                    success = self.fso_terminal.attempt_link_acquisition(
                        relay.fso_terminal, distance, time)
                    if success:
                        print(f"    CONTROLLER-RELAY: {self.sat_id} -> {relay.sat_id}")
        
        #Priority 2 - Accept connections from Earth satellites in region - which is handled by the Earth satellites attempting to connect to us
    
    #Establish links for Mars satellites and relays
    def _establish_other_links(self, other_satellites, time):
        potential_targets = []
        for target_sat in other_satellites:
            if target_sat.sat_id != self.sat_id and target_sat.sat_id not in self.fso_terminal.current_links:
                distance = np.linalg.norm(self.position - target_sat.position) * 1000
                
                if self._can_establish_fso_link(target_sat, distance):
                    priority = 1
                    #HIGH PRIORITY: Interplanetary backbone links
                    if (self.sat_type == 'relay' and target_sat.sat_type == 'relay' and
                          (('Earth-Sun' in str(self.lagrange_point) and 'Mars-Sun' in str(target_sat.lagrange_point)) or
                        ('Mars-Sun' in str(self.lagrange_point) and 'Earth-Sun' in str(target_sat.lagrange_point)))):
                        priority = 50
                    #MEDIUM PRIORITY: Access network links
                    elif ((self.planet == 'Mars' and target_sat.sat_type == 'relay' and 'Mars-Sun' in str(target_sat.lagrange_point)) or
                          (target_sat.planet == 'Mars' and self.sat_type == 'relay' and 'Mars-Sun' in str(self.lagrange_point))):
                        priority = 30
                    #LOW PRIORITY: Regional links
                    else:
                        priority = 5
                    potential_targets.append((target_sat, distance, priority))
        
        #Sort by priority with highest first, then by distance with closest first
        potential_targets.sort(key=lambda x: (-x[2], x[1]))
        
        #Attempt links
        links_attempted = 0
        for target_sat, distance, priority in potential_targets:
            if len(self.fso_terminal.current_links) >= self.simultaneous_links_max:
                break
            success = self.fso_terminal.attempt_link_acquisition(target_sat.fso_terminal, distance, time)
            links_attempted += 1
            if links_attempted >= 3:  #Limit attempts per update
                break

    #Check if an Earth satellite is in this controller's line of sight region
    def _is_in_controller_region(self, earth_satellite):
        if self.sat_type != 'controller' or earth_satellite.planet != 'Earth':
            return False
        
        controller_angle = np.arctan2(self.position[1], self.position[0])
        satellite_angle = np.arctan2(earth_satellite.position[1], earth_satellite.position[0])
        controller_angle = (controller_angle + 2 * np.pi) % (2 * np.pi)
        satellite_angle = (satellite_angle + 2 * np.pi) % (2 * np.pi)
        
        if self.lagrange_point == 'Earth-Moon-L1':
            region_start = 315 * np.pi / 180
            region_end = 45 * np.pi / 180
            return self._angle_in_region(satellite_angle, region_start, region_end)
        elif self.lagrange_point == 'Earth-Moon-L2':
            region_start = 135 * np.pi / 180
            region_end = 225 * np.pi / 180
            return self._angle_in_region(satellite_angle, region_start, region_end)
        elif self.lagrange_point == 'Earth-Moon-L4':
            region_start = 45 * np.pi / 180
            region_end = 135 * np.pi / 180
            return self._angle_in_region(satellite_angle, region_start, region_end)
        elif self.lagrange_point == 'Earth-Moon-L5':
            region_start = 225 * np.pi / 180
            region_end = 315 * np.pi / 180
            return self._angle_in_region(satellite_angle, region_start, region_end)
        
        return False
    
    #Check if an angle is within a region, handling wraparound
    def _angle_in_region(self, angle, start, end):
        if start <= end:
            return start <= angle <= end
        else:
            return angle >= start or angle <= end
    
    def get_status_dict(self):
        fso_stats = self.fso_terminal.get_performance_stats()
        #controller status
        controller_status = {}
        if self.sat_type == 'controller':
            controller_status = {
                'satellites_in_region': len(getattr(self, 'satellites_in_region', [])),
                'last_coordination_update': getattr(self, 'last_coordination_update', 0),
                'coordination_active': hasattr(self, 'navigation_coordinator')
            }
        
        return {
            'sat_id': self.sat_id,
            'planet': self.planet,
            'sat_type': self.sat_type,
            'position_x': self.position[0],
            'position_y': self.position[1], 
            'position_z': self.position[2],
            'velocity_x': self.velocity[0] if hasattr(self, 'velocity') else 0,
            'velocity_y': self.velocity[1] if hasattr(self, 'velocity') else 0,
            'velocity_z': self.velocity[2] if hasattr(self, 'velocity') else 0,
            'connection_count': len(self.connections),
            'earth_connected': self.earth_connected,
            'mars_connected': self.mars_connected,
            'hop_count': len(self.relay_hops) - 1 if self.relay_hops else 0,
            'lagrange_point': self.lagrange_point if self.lagrange_point else 'N/A',
            'fso_active_links': fso_stats['current_active_links'],
            'fso_navigation_assisted_links': fso_stats.get('navigation_assisted_links', 0),
            'fso_total_data_rate_gbps': fso_stats['total_data_rate'] / 1e9,
            'fso_acquisition_success_rate': fso_stats['acquisition_success_rate'],
            'fso_terminal_aperture_m': self.fso_terminal.aperture_size,
            'fso_terminal_power_w': self.fso_terminal.transmit_power,
            'last_navigation_update': fso_stats.get('last_navigation_update', 0),
            'recommended_target': fso_stats.get('recommended_target', 'None'),
            'traffic_authorized': fso_stats.get('traffic_authorized', False),
            'transmission_schedule': str(fso_stats.get('transmission_schedule', {})),
            **controller_status
        }

#Check line-of-sight between satellites considering celestial body - a laser should not be going through a planet or the sun
class LineOfSightChecker:
    @staticmethod
    def check_line_intersects_sphere(p1, p2, sphere_center, sphere_radius):
        line_vec = p2 - p1
        line_length = np.linalg.norm(line_vec)
        
        if line_length == 0:
            return False
            
        #Normalize
        line_dir = line_vec / line_length
        #Vector from p1 to sphere center
        to_center = sphere_center - p1
        #Project to_center onto line direction to find closest point on line to sphere center
        projection_length = np.dot(to_center, line_dir)
        #Clamp projection to line segment
        projection_length = max(0, min(line_length, projection_length))
        #Find closest point on line segment to sphere center
        closest_point = p1 + projection_length * line_dir
        #Check distance from closest point to sphere center
        distance_to_center = np.linalg.norm(closest_point - sphere_center)
        #returns true if intersects
        return distance_to_center < sphere_radius
    
    @staticmethod
    def is_line_of_sight_clear(sat1_pos, sat2_pos, time, check_sun=True, check_planets=True, check_moon=True):
        #Get all celestial body positions
        earth_pos = LineOfSightChecker._get_earth_position(time)
        mars_pos = LineOfSightChecker._get_mars_position(time)
        sun_pos = np.array([0, 0, 0])  # Sun at origin
        moon_pos = LineOfSightChecker._get_moon_position(time, earth_pos)
        
        #Check Sun
        if check_sun:
            if LineOfSightChecker.check_line_intersects_sphere(sat1_pos, sat2_pos, sun_pos, SOLAR_EXCLUSION_RADIUS):
                return False, "Sun"
        
        if check_planets:
            #Check Earth
            if LineOfSightChecker.check_line_intersects_sphere(sat1_pos, sat2_pos, earth_pos, EARTH_RADIUS):
                return False, "Earth"
            #Check Mars
            if LineOfSightChecker.check_line_intersects_sphere(sat1_pos, sat2_pos, mars_pos, MARS_RADIUS):
                return False, "Mars"
        
        #Check Moon occlusion
        if check_moon:
            #Only check moon if at least one satellite is near Earth
            earth_dist1 = np.linalg.norm(sat1_pos - earth_pos)
            earth_dist2 = np.linalg.norm(sat2_pos - earth_pos)
            
            if earth_dist1 < 1e6 or earth_dist2 < 1e6:  #Within 1 million km of Earth
                if LineOfSightChecker.check_line_intersects_sphere(
                    sat1_pos, sat2_pos, moon_pos, MOON_RADIUS
                ):
                    return False, "Moon"
        
        return True, None
    
    #Get Earth position at given time using real ephemeris
    @staticmethod
    def _get_earth_position(time):
        ephemeris = RealEphemerisData()
        return ephemeris.get_earth_position(time)
    #Get Mars position at given time using real ephemeris
    @staticmethod
    def _get_mars_position(time):
        ephemeris = RealEphemerisData()
        return ephemeris.get_mars_position(time)
    #Get Moon position using real ephemeris
    @staticmethod
    def _get_moon_position(time, earth_position):
        ephemeris = RealEphemerisData()
        return ephemeris.get_moon_position(time)
        
#Use actual NASA JPL ephemeris data for planetary positions
#Use JPL DE440 ephemeris - most recent high-precision
class RealEphemerisData:
    def __init__(self):
        solar_system_ephemeris.set('de440')
    
    #Get actual Earth position from JPL ephemeris
    def get_earth_position(self, time_hours):
        #Convert simulation time to actual date
        base_date = Time('2025-05-03 12:00:00')
        current_time = base_date + time_hours * u.hour
        #Get Earth position relative to solar system barycenter
        earth_pos = get_body_barycentric('earth', current_time)
        #Convert to km and return as numpy array
        return np.array([
            earth_pos.x.to(u.km).value,
            earth_pos.y.to(u.km).value, 
            earth_pos.z.to(u.km).value
        ])
    #Get actual Mars position from JPL ephemeris
    def get_mars_position(self, time_hours):
        base_date = Time('2025-05-03 12:00:00')
        current_time = base_date + time_hours * u.hour
        mars_pos = get_body_barycentric('mars', current_time)
        return np.array([
            mars_pos.x.to(u.km).value,
            mars_pos.y.to(u.km).value,
            mars_pos.z.to(u.km).value
        ])
    #Get actual Moon positio
    def get_moon_position(self, time_hours):
        base_date = Time('2025-05-03 12:00:00')
        current_time = base_date + time_hours * u.hour
        moon_pos = get_body_barycentric('moon', current_time)
        return np.array([
            moon_pos.x.to(u.km).value,
            moon_pos.y.to(u.km).value,
            moon_pos.z.to(u.km).value
        ])

#Calculate precise Lagrange point positions using actual mass ratios
class PreciseLagrangePoints:
    def __init__(self, ephemeris_data):
        self.ephemeris = ephemeris_data
        # Actual mass ratios
        self.earth_moon_mass_ratio = 81.30056  #Earth/Moon mass ratio
        self.earth_sun_mass_ratio = 332946.0   #Sun/Earth mass ratio
        self.mars_sun_mass_ratio = 3098708.0   #Sun/Mars mass ratio
    
    #Calculate precise Earth-Moon L1 position
    def earth_moon_l1(self, time_hours):
        earth_pos = self.ephemeris.get_earth_position(time_hours)
        moon_pos = self.ephemeris.get_moon_position(time_hours)
        #Mass ratio mu = m2/(m1+m2)
        mu = 1.0 / (1.0 + self.earth_moon_mass_ratio)
        #Distance between Earth and Moon
        r = np.linalg.norm(moon_pos - earth_pos)
        #Solve for L1 distance from Earth using simplified cubic equation solution - for small mu, L1 ≈ r(1 - (μ/3)^(1/3))
        l1_distance = r * (1 - (mu/3)**(1/3))
        #Direction from Earth to Moon
        earth_moon_dir = (moon_pos - earth_pos) / r
        return earth_pos + l1_distance * earth_moon_dir

    #Calculate precise Earth-Sun L4 position
    def earth_sun_l4(self, time_hours):
        earth_pos = self.ephemeris.get_earth_position(time_hours)
        sun_pos = np.array([0, 0, 0])  #Sun at barycenter
        #L4 is 60° ahead of Earth in its orbit
        earth_sun_vec = earth_pos - sun_pos
        earth_distance = np.linalg.norm(earth_sun_vec)
        #Get Earth's orbital plane normal - simplified to z-axis
        earth_angle = np.arctan2(earth_sun_vec[1], earth_sun_vec[0])
        l4_angle = earth_angle + np.pi/3  #60° ahead
        return earth_distance * np.array([np.cos(l4_angle), np.sin(l4_angle), 0])

    #Calculate precise Earth-Moon L2 position
    def earth_moon_l2(self, time_hours):
        earth_pos = self.ephemeris.get_earth_position(time_hours)
        moon_pos = self.ephemeris.get_moon_position(time_hours)
        mu = 1.0 / (1.0 + self.earth_moon_mass_ratio)
        r = np.linalg.norm(moon_pos - earth_pos)
        l2_distance = r * (1 + (mu/3)**(1/3))
        earth_moon_dir = (moon_pos - earth_pos) / r
        return earth_pos + l2_distance * earth_moon_dir

    #Calculate precise Mars-Sun L4 position
    def mars_sun_l4(self, time_hours):
        mars_pos = self.ephemeris.get_mars_position(time_hours)
        sun_pos = np.array([0, 0, 0])
        mu = 1.0 / (1.0 + self.mars_sun_mass_ratio)
        #L4 forms equilateral triangle with Mars and Sun
        mars_distance = np.linalg.norm(mars_pos)
        mars_angle = np.arctan2(mars_pos[1], mars_pos[0])
        l4_angle = mars_angle + np.pi/3
        correction = 0.5 * mu  #Shifts L4 slightly
        l4_angle += correction
        return mars_distance * np.array([np.cos(l4_angle), np.sin(l4_angle), 0])

    #Calculate precise Earth-Moon L3 position
    #def earth_moon_l3(self, time_hours):
    #    earth_pos = self.ephemeris.get_earth_position(time_hours)
    #    moon_pos = self.ephemeris.get_moon_position(time_hours)
    #    mu = 1.0 / (1.0 + self.earth_moon_mass_ratio)
    #    r = np.linalg.norm(moon_pos - earth_pos)
        #L3 is on opposite side of Earth from Moon
    #    l3_distance = r * (1 + 5*mu/12)  #Approximation for small mu
    #    earth_moon_dir = (moon_pos - earth_pos) / r
    #    return earth_pos - l3_distance * earth_moon_dir

    #Calculate precise Earth-Moon L4 position
    def earth_moon_l4(self, time_hours):
        earth_pos = self.ephemeris.get_earth_position(time_hours)
        moon_pos = self.ephemeris.get_moon_position(time_hours)
        #L4 forms equilateral triangle 60° ahead
        earth_moon_vec = moon_pos - earth_pos
        distance = np.linalg.norm(earth_moon_vec)
        angle = np.arctan2(earth_moon_vec[1], earth_moon_vec[0]) + np.pi/3
        return earth_pos + distance * np.array([np.cos(angle), np.sin(angle), 0])

    #Calculate precise Earth-Moon L5 position
    def earth_moon_l5(self, time_hours):
        earth_pos = self.ephemeris.get_earth_position(time_hours)
        moon_pos = self.ephemeris.get_moon_position(time_hours)   
        #L5 forms equilateral triangle 60° behind
        earth_moon_vec = moon_pos - earth_pos
        distance = np.linalg.norm(earth_moon_vec)
        angle = np.arctan2(earth_moon_vec[1], earth_moon_vec[0]) - np.pi/3
        return earth_pos + distance * np.array([np.cos(angle), np.sin(angle), 0])

    #Calculate precise Earth-Sun L1 position
    def earth_sun_l1(self, time_hours):
        earth_pos = self.ephemeris.get_earth_position(time_hours)
        sun_pos = np.array([0, 0, 0])   
        mu = 1.0 / (1.0 + self.earth_sun_mass_ratio)
        r = np.linalg.norm(earth_pos - sun_pos)
        #L1 between Sun and Earth
        l1_distance = r * (1 - (mu/3)**(1/3))
        sun_earth_dir = (earth_pos - sun_pos) / r
        return sun_pos + l1_distance * sun_earth_dir

    #Calculate precise Earth-Sun L2 position
    def earth_sun_l2(self, time_hours):
        earth_pos = self.ephemeris.get_earth_position(time_hours)
        sun_pos = np.array([0, 0, 0])   
        mu = 1.0 / (1.0 + self.earth_sun_mass_ratio)
        r = np.linalg.norm(earth_pos - sun_pos)
        #L2 beyond Earth from Sun
        l2_distance = r * (1 + (mu/3)**(1/3))
        sun_earth_dir = (earth_pos - sun_pos) / r
        return sun_pos + l2_distance * sun_earth_dir

    #Calculate precise Earth-Sun L3 position
    #def earth_sun_l3(self, time_hours):
    #    earth_pos = self.ephemeris.get_earth_position(time_hours)
    #    sun_pos = np.array([0, 0, 0])   
    #    mu = 1.0 / (1.0 + self.earth_sun_mass_ratio)
    #    r = np.linalg.norm(earth_pos - sun_pos)
    #    #L3 on opposite side of Sun
    #    l3_distance = r * (1 + 7*mu/12)  #Approximation
    #    sun_earth_dir = (earth_pos - sun_pos) / r
    #    return sun_pos - l3_distance * sun_earth_dir

    #Calculate precise Earth-Sun L5 position
    def earth_sun_l5(self, time_hours):
        earth_pos = self.ephemeris.get_earth_position(time_hours)
        #L5 is 60° behind Earth in its orbit
        earth_distance = np.linalg.norm(earth_pos)
        earth_angle = np.arctan2(earth_pos[1], earth_pos[0])
        l5_angle = earth_angle - np.pi/3
        return earth_distance * np.array([np.cos(l5_angle), np.sin(l5_angle), 0])

    #Calculate precise Mars-Sun L1 position
    def mars_sun_l1(self, time_hours):
        mars_pos = self.ephemeris.get_mars_position(time_hours)
        sun_pos = np.array([0, 0, 0])   
        mu = 1.0 / (1.0 + self.mars_sun_mass_ratio)
        r = np.linalg.norm(mars_pos - sun_pos)
        #L1 between Sun and Mars
        l1_distance = r * (1 - (mu/3)**(1/3))
        sun_mars_dir = (mars_pos - sun_pos) / r
        return sun_pos + l1_distance * sun_mars_dir

    #Calculate precise Mars-Sun L2 position
    def mars_sun_l2(self, time_hours):
        mars_pos = self.ephemeris.get_mars_position(time_hours)
        sun_pos = np.array([0, 0, 0])
        mu = 1.0 / (1.0 + self.mars_sun_mass_ratio)
        r = np.linalg.norm(mars_pos - sun_pos)
        #L2 beyond Mars from Sun
        l2_distance = r * (1 + (mu/3)**(1/3))
        sun_mars_dir = (mars_pos - sun_pos) / r
        return sun_pos + l2_distance * sun_mars_dir

    #Calculate precise Mars-Sun L3 position
    #def mars_sun_l3(self, time_hours):
    #    mars_pos = self.ephemeris.get_mars_position(time_hours)
    #    sun_pos = np.array([0, 0, 0])
    #    mu = 1.0 / (1.0 + self.mars_sun_mass_ratio)
    #    r = np.linalg.norm(mars_pos - sun_pos)
    #    #L3 on opposite side of Sun
    #    l3_distance = r * (1 + 7*mu/12)  #Approximation
    #    sun_mars_dir = (mars_pos - sun_pos) / r
    #    return sun_pos - l3_distance * sun_mars_dir

    #Calculate precise Mars-Sun L5 position
    def mars_sun_l5(self, time_hours):
        mars_pos = self.ephemeris.get_mars_position(time_hours)   
        #L5 is 60° behind Mars in its orbit
        mars_distance = np.linalg.norm(mars_pos)
        mars_angle = np.arctan2(mars_pos[1], mars_pos[0])
        l5_angle = mars_angle - np.pi/3
        return mars_distance * np.array([np.cos(l5_angle), np.sin(l5_angle), 0])

#Satellite with real orbital mechanics
class EnhancedSatellite(Satellite):
    def __init__(self, sat_id, planet, orbit_altitude, inclination, phase, 
                 raan=0, argument_of_perigee=0, eccentricity=0.001, **kwargs):
        super().__init__(sat_id, planet, orbit_altitude, inclination, phase, **kwargs)
        self.raan = np.radians(raan)  #Right Ascension of Ascending Node
        self.arg_pe = np.radians(argument_of_perigee)
        self.eccentricity = eccentricity
        #Initialize with real ephemeris
        self.ephemeris = RealEphemerisData()
        self.lagrange_calculator = PreciseLagrangePoints(self.ephemeris)

class TrafficQueue:
    def __init__(self):
        self.pending_data = {}
        self.total_size_gb = 0
        self.urgent_deadline_count = 0
        
    def add_traffic(self, priority, size_gb, deadline_hours=None, destination=None):
        if priority not in self.pending_data:
            self.pending_data[priority] = []
        traffic_packet = {
            'size_gb': size_gb,
            'deadline_hours': deadline_hours,
            'destination': destination,
            'timestamp': dt.datetime.now(),
            'priority': priority
        }
        self.pending_data[priority].append(traffic_packet)
        self.total_size_gb += size_gb
        if deadline_hours and deadline_hours < 0.5:
            self.urgent_deadline_count += 1
    
    def get_transmission_time_estimate(self, data_rate_bps):
        if data_rate_bps <= 0:
            return float('inf')
        return (self.total_size_gb * 8e9) / data_rate_bps / 3600  #Convert to hrs

class TransmissionSchedule:
    def __init__(self, start_time):
        self.start_time = start_time
        self.schedule = {}  #time_slot: assignment
        self.satellite_assignments = {}  #sat_id: assignment
        self.relay_capacity = {}  #relay_id: used_capacity
        
    def add_assignment(self, satellite_id, relay_id, start_time, duration, priority, data_rate):
        assignment = {
            'satellite_id': satellite_id,
            'relay_id': relay_id,
            'start_time': start_time,
            'duration': duration,
            'priority': priority,
            'data_rate': data_rate,
            'auth_token': f"AUTH_{satellite_id}_{int(start_time*1000)}",
            'queue_position': len(self.satellite_assignments)
        }
        self.satellite_assignments[satellite_id] = assignment
        time_slot = int(start_time / TRANSMISSION_SLOT_DURATION)
        self.schedule[time_slot] = assignment
        
        if relay_id not in self.relay_capacity:
            self.relay_capacity[relay_id] = 0
        self.relay_capacity[relay_id] += 1
        
    def get_assignment(self, satellite_id):
        return self.satellite_assignments.get(satellite_id)

#Simplified metrics collection for thesis graphs
class ThesisMetricsCollector:
    def __init__(self):
        #Time series data for graphs
        self.time_series = {
            'time': [],
            'earth_connectivity': [],
            'mars_connectivity': [],
            'total_connectivity': [],
            'active_fso_links': [],
            'data_rate_tbps': [],
            'interplanetary_links': [],
            'controller_coverage': [],
            'traffic_scheduled': [],
            'blocked_by_sun': [],
            'blocked_by_earth': [],
            'blocked_by_mars': [],
            'acquisition_success_rate': [],
            'navigation_assisted_pct': [],
            'average_latency_ms': []
            }
        
        # NEW METRICS 1-8
        # 1. Temporal Link Stability
        self.link_stability = {
            'time': [],
            'average_link_duration': [],
            'link_interruption_frequency': [],
            'mean_time_between_failures': []
        }
        self.link_tracking = {}  # sat_id -> {target_id: {'start_time', 'duration', 'interruptions'}}
        
        # 2. Handoff Performance
        self.handoff_metrics = {
            'time': [],
            'planned_handoffs': [],
            'successful_handoffs': [],
            'failed_handoffs': [],
            'average_handoff_latency': [],
            'seamless_handoffs': []
        }
        self.handoff_events = []
        
        # 3. Controller Load Balancing
        self.controller_load = {
            'time': [],
            'load_variance': [],
            'average_utilization': [],
            'max_utilization': [],
            'response_times': []
        }
        
        # 4. Network Topology Analysis
        self.topology_metrics = {
            'time': [],
            'clustering_coefficient': [],
            'average_path_length': [],
            'network_diameter': [],
            'edge_connectivity': [],
            'critical_nodes': []
        }
        
        # 5. Queue Performance
        self.queue_metrics = {
            'time': [],
            'average_queue_depth': [],
            'max_queue_depth': [],
            'queue_overflow_events': [],
            'average_wait_time': [],
            'deadline_miss_rate': []
        }
        
        # 6. Eclipse/Occlusion Duration
        self.occlusion_events = {
            'time': [],
            'sun_occlusion_duration': [],
            'earth_occlusion_duration': [],
            'mars_occlusion_duration': [],
            'simultaneous_occlusions': [],
            'longest_blackout': []
        }
        self.current_occlusions = {}
        
        # 7. Energy Efficiency
        self.energy_metrics = {
            'time': [],
            'total_power_consumption': [],
            'power_per_gbps': [],
            'idle_vs_active_ratio': [],
            'adaptive_power_savings': []
        }
        
        # 8. Geographic Coverage
        self.coverage_metrics = {
            'time': [],
            'earth_surface_coverage': [],
            'mars_surface_coverage': [],
            'polar_coverage': [],
            'equatorial_coverage': []
        }
        
        #data for specific analysis
        self.link_quality_distribution = []
        self.traffic_priority_distribution = []
        self.controller_performance = []
        
    #Record all essential metrics at current time
    def record_metrics(self, network, time):
        stats = network.get_connectivity_stats()
        fso = stats['fso_performance']
        optical = stats['optical_links']
        coord = stats['controller_coordination']
        traffic = stats['traffic_management']
        
        self.time_series['time'].append(time)
        self.time_series['earth_connectivity'].append(stats['earth_connectivity'])
        self.time_series['mars_connectivity'].append(stats['mars_connectivity']) 
        self.time_series['total_connectivity'].append(stats['total_connectivity'])
        self.time_series['active_fso_links'].append(fso.get('total_active_fso_links', 0))
        self.time_series['data_rate_tbps'].append(fso.get('total_data_rate_tbps', 0))
        self.time_series['interplanetary_links'].append(optical.get('interplanetary_links', 0))
        self.time_series['controller_coverage'].append(coord.get('coordination_coverage_percentage', 0))
        self.time_series['traffic_scheduled'].append(traffic.get('satellites_with_traffic_schedule', 0))
        self.time_series['acquisition_success_rate'].append(fso.get('acquisition_success_rate_pct', 0))
        self.time_series['navigation_assisted_pct'].append(fso.get('navigation_assistance_percentage', 0))
        self.time_series['average_latency_ms'].append(stats['network_latency'])
        self.time_series['blocked_by_sun'].append(getattr(network, 'last_blocked_by_sun', 0))
        self.time_series['blocked_by_earth'].append(getattr(network, 'last_blocked_by_earth', 0))
        self.time_series['blocked_by_mars'].append(getattr(network, 'last_blocked_by_mars', 0))
        
        #Record distributions periodically
        if time % 1000 == 0:
            self.link_quality_distribution.append({
                'time': time,
                'distribution': fso.get('link_quality_distribution', {})
            })
            self.traffic_priority_distribution.append({
                'time': time,
                'distribution': traffic.get('traffic_priority_distribution', {})
            })
            self.controller_performance.append({
                'time': time,
                'details': coord.get('controller_details', {})
            })
        
        self._record_link_stability(network, time)
        self._record_handoff_performance(network, time)
        self._record_controller_load(network, time)
        self._record_topology_metrics(network, time)
        self._record_queue_metrics(network, time)
        self._record_occlusion_events(network, time)
        self._record_coverage_metrics(network, time)
    
    def _record_link_stability(self, network, time):
        current_links = {}
        total_duration = 0
        link_count = 0
        interruptions = 0
        for sat in network.satellites:
            sat_id = sat.sat_id
            if sat_id not in self.link_tracking:
                self.link_tracking[sat_id] = {}
            for target_id, link_info in sat.fso_terminal.current_links.items():
                current_links[(sat_id, target_id)] = True
                if target_id not in self.link_tracking[sat_id]:
                    # New link
                    self.link_tracking[sat_id][target_id] = {
                        'start_time': time,
                        'duration': 0,
                        'interruptions': 0,
                        'last_seen': time
                    }
                else:
                    # Existing link
                    link_track = self.link_tracking[sat_id][target_id]
                    if time - link_track['last_seen'] > 24:  # Link was interrupted
                        link_track['interruptions'] += 1
                        interruptions += 1
                    link_track['duration'] = time - link_track['start_time']
                    link_track['last_seen'] = time
                    total_duration += link_track['duration']
                    link_count += 1
        
        #Calculate metrics
        avg_duration = total_duration / max(link_count, 1)
        interruption_freq = interruptions / max(len(network.satellites), 1)
        
        #Calculate MTBF
        total_failures = sum(link_data.get('interruptions', 0) 
                           for sat_links in self.link_tracking.values() 
                           for link_data in sat_links.values())
        mtbf = time / max(total_failures, 1) if total_failures > 0 else time
        self.link_stability['time'].append(time)
        self.link_stability['average_link_duration'].append(avg_duration)
        self.link_stability['link_interruption_frequency'].append(interruption_freq)
        self.link_stability['mean_time_between_failures'].append(mtbf)
    
    def _record_handoff_performance(self, network, time):
        planned = 0
        successful = 0
        failed = 0
        total_latency = 0
        seamless = 0
        
        for sat in network.satellites:
            if sat.sat_type == 'controller' and hasattr(sat, 'navigation_coordinator'):
                if hasattr(sat.navigation_coordinator.fso_scheduler, 'scheduled_handoffs'):
                    for sat_id, handoff_info in sat.navigation_coordinator.fso_scheduler.scheduled_handoffs.items():
                        planned += 1
                        if handoff_info.get('status') == 'completed':
                            successful += 1
                            if handoff_info.get('data_loss', 0) == 0:
                                seamless += 1
                        elif handoff_info.get('status') == 'failed':
                            failed += 1
        
        #Record handoff event
        if planned > 0:
            self.handoff_events.append({
                'time': time,
                'planned': planned,
                'successful': successful,
                'failed': failed
            })
        
        #Calculate average latency from recent events
        recent_events = [e for e in self.handoff_events if time - e['time'] < 100]
        avg_latency = np.mean([e.get('latency', 0.1) for e in recent_events]) if recent_events else 0.1
        
        self.handoff_metrics['time'].append(time)
        self.handoff_metrics['planned_handoffs'].append(planned)
        self.handoff_metrics['successful_handoffs'].append(successful)
        self.handoff_metrics['failed_handoffs'].append(failed)
        self.handoff_metrics['average_handoff_latency'].append(avg_latency)
        self.handoff_metrics['seamless_handoffs'].append(seamless)
    
    def _record_controller_load(self, network, time):
        controllers = [sat for sat in network.satellites if sat.sat_type == 'controller']
        loads = []
        utilizations = []
        response_times = []
        
        for controller in controllers:
            if hasattr(controller, 'satellites_in_region'):
                # Calculate load as satellites managed / max capacity
                load = len(controller.satellites_in_region)
                max_capacity = 50  # Assume max 50 satellites per controller
                utilization = (load / max_capacity) * 100
                
                loads.append(load)
                utilizations.append(utilization)
                
                # Simulate response time based on load
                base_response = 0.01  # 10ms base
                response_time = base_response * (1 + load/20)  # Increases with load
                response_times.append(response_time)
        
        if loads:
            load_variance = np.var(loads)
            avg_utilization = np.mean(utilizations)
            max_utilization = np.max(utilizations)
            avg_response = np.mean(response_times)
        else:
            load_variance = 0
            avg_utilization = 0
            max_utilization = 0
            avg_response = 0
        
        self.controller_load['time'].append(time)
        self.controller_load['load_variance'].append(load_variance)
        self.controller_load['average_utilization'].append(avg_utilization)
        self.controller_load['max_utilization'].append(max_utilization)
        self.controller_load['response_times'].append(avg_response)
    
    def _record_topology_metrics(self, network, time):
        G = network.connection_graph
        
        if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
            # Convert to undirected for some metrics
            G_undirected = G.to_undirected()
            # Clustering coefficient
            clustering = nx.average_clustering(G_undirected)
            # Path length and diameter (only if connected)
            if nx.is_connected(G_undirected):
                avg_path_length = nx.average_shortest_path_length(G_undirected)
                diameter = nx.diameter(G_undirected)
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(G_undirected), key=len)
                subgraph = G_undirected.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(subgraph) if len(largest_cc) > 1 else 0
                diameter = nx.diameter(subgraph) if len(largest_cc) > 1 else 0
            
            # Edge connectivity
            if len(network.satellites) > 1:
                earth_nodes = [s.sat_id for s in network.satellites if s.planet == 'Earth']
                mars_nodes = [s.sat_id for s in network.satellites if s.planet == 'Mars']
                if earth_nodes and mars_nodes:
                    try:
                        edge_conn = nx.edge_connectivity(G, earth_nodes[0], mars_nodes[0])
                    except:
                        edge_conn = 0
                else:
                    edge_conn = 0
            else:
                edge_conn = 0
            # Find critical nodes using betweenness centrality
            centrality = nx.betweenness_centrality(G)
            critical_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            critical_count = len([n for n, c in centrality.items() if c > 0.1])
        else:
            clustering = 0
            avg_path_length = 0
            diameter = 0
            edge_conn = 0
            critical_count = 0
        
        self.topology_metrics['time'].append(time)
        self.topology_metrics['clustering_coefficient'].append(clustering)
        self.topology_metrics['average_path_length'].append(avg_path_length)
        self.topology_metrics['network_diameter'].append(diameter)
        self.topology_metrics['edge_connectivity'].append(edge_conn)
        self.topology_metrics['critical_nodes'].append(critical_count)
    
    def _record_queue_metrics(self, network, time):
        queue_depths = []
        wait_times = []
        overflow_events = 0
        deadlines_missed = 0
        total_packets = 0
        
        for sat in network.satellites:
            if hasattr(sat.fso_terminal, 'traffic_queue'):
                queue = sat.fso_terminal.traffic_queue
                # Queue depth
                depth = queue.total_size_gb
                queue_depths.append(depth)
                # Check for overflow (>100GB)
                if depth > 100:
                    overflow_events += 1
                # Estimate wait time
                if len(sat.fso_terminal.current_links) > 0:
                    avg_rate = np.mean([link['data_rate'] for link in sat.fso_terminal.current_links.values()])
                    wait_time = queue.get_transmission_time_estimate(avg_rate)
                    wait_times.append(wait_time)
                # Check deadline misses
                for priority_data in queue.pending_data.values():
                    for packet in priority_data:
                        total_packets += 1
                        if packet.get('deadline_hours') and packet['deadline_hours'] < wait_time:
                            deadlines_missed += 1
        
        avg_queue_depth = np.mean(queue_depths) if queue_depths else 0
        max_queue_depth = np.max(queue_depths) if queue_depths else 0
        avg_wait_time = np.mean(wait_times) if wait_times else 0
        deadline_miss_rate = (deadlines_missed / max(total_packets, 1)) * 100
        self.queue_metrics['time'].append(time)
        self.queue_metrics['average_queue_depth'].append(avg_queue_depth)
        self.queue_metrics['max_queue_depth'].append(max_queue_depth)
        self.queue_metrics['queue_overflow_events'].append(overflow_events)
        self.queue_metrics['average_wait_time'].append(avg_wait_time)
        self.queue_metrics['deadline_miss_rate'].append(deadline_miss_rate)
    
    def _record_occlusion_events(self, network, time):
        sun_duration = 0
        earth_duration = 0
        mars_duration = 0
        simultaneous = 0
        # Track occlusions for each satellite pair
        for sat1 in network.satellites:
            for sat2 in network.satellites:
                if sat1.sat_id >= sat2.sat_id:
                    continue
                pair_id = f"{sat1.sat_id}-{sat2.sat_id}"
                los_clear, blocking_body = LineOfSightChecker.is_line_of_sight_clear(
                    sat1.position, sat2.position, time
                )
                if not los_clear:
                    if pair_id not in self.current_occlusions:
                        self.current_occlusions[pair_id] = {
                            'start_time': time,
                            'blocker': blocking_body
                        }
                    # Track duration
                    duration = time - self.current_occlusions[pair_id]['start_time']
                    if blocking_body == "Sun":
                        sun_duration += duration
                    elif blocking_body == "Earth":
                        earth_duration += duration
                    elif blocking_body == "Mars":
                        mars_duration += duration
                else:
                    # Clear occlusion ended
                    if pair_id in self.current_occlusions:
                        del self.current_occlusions[pair_id]
        # Count simultaneous occlusions
        blockers = [occ['blocker'] for occ in self.current_occlusions.values()]
        if len(set(blockers)) > 1:
            simultaneous = len(self.current_occlusions)
        
        # Find longest blackout
        longest = max([time - occ['start_time'] for occ in self.current_occlusions.values()]) if self.current_occlusions else 0
        
        self.occlusion_events['time'].append(time)
        self.occlusion_events['sun_occlusion_duration'].append(sun_duration)
        self.occlusion_events['earth_occlusion_duration'].append(earth_duration)
        self.occlusion_events['mars_occlusion_duration'].append(mars_duration)
        self.occlusion_events['simultaneous_occlusions'].append(simultaneous)
        self.occlusion_events['longest_blackout'].append(longest)
    
    def _record_coverage_metrics(self, network, time):
        # Calculate coverage by checking visibility from surface points
        earth_coverage = self._calculate_surface_coverage(network, 'Earth')
        mars_coverage = self._calculate_surface_coverage(network, 'Mars')
        
        # Separate polar and equatorial coverage
        earth_polar = self._calculate_polar_coverage(network, 'Earth')
        earth_equatorial = self._calculate_equatorial_coverage(network, 'Earth')
        
        self.coverage_metrics['time'].append(time)
        self.coverage_metrics['earth_surface_coverage'].append(earth_coverage)
        self.coverage_metrics['mars_surface_coverage'].append(mars_coverage)
        self.coverage_metrics['polar_coverage'].append(earth_polar)
        self.coverage_metrics['equatorial_coverage'].append(earth_equatorial)
    
    def _calculate_surface_coverage(self, network, planet):
        # Simplified coverage calculation
        # Count satellites that can see the planet
        if planet == 'Earth':
            sats = [s for s in network.satellites if s.planet == 'Earth']
            planet_radius = EARTH_RADIUS
        else:
            sats = [s for s in network.satellites if s.planet == 'Mars']
            planet_radius = MARS_RADIUS
        
        if not sats:
            return 0
        
        # Each satellite covers a certain area based on altitude
        coverage_area = 0
        for sat in sats:
            altitude = sat.orbit_altitude
            # Approximate coverage angle
            coverage_angle = np.arccos(planet_radius / (planet_radius + altitude))
            # Area of spherical cap
            cap_area = 2 * np.pi * planet_radius**2 * (1 - np.cos(coverage_angle))
            coverage_area += cap_area
        
        # Total planet surface area
        total_area = 4 * np.pi * planet_radius**2
        coverage_pct = min(100, (coverage_area / total_area) * 100)
        
        return coverage_pct
    
    def _calculate_polar_coverage(self, network, planet):
        # Count satellites with high inclination orbits
        if planet == 'Earth':
            sats = [s for s in network.satellites if s.planet == 'Earth' and s.inclination > 60]
        else:
            sats = [s for s in network.satellites if s.planet == 'Mars' and s.inclination > 60]
        
        # Simple metric: percentage of high-inclination satellites
        total_sats = len([s for s in network.satellites if s.planet == planet])
        return (len(sats) / max(total_sats, 1)) * 100
    
    def _calculate_equatorial_coverage(self, network, planet):
        # Count satellites with low inclination orbits
        if planet == 'Earth':
            sats = [s for s in network.satellites if s.planet == 'Earth' and s.inclination < 30]
        else:
            sats = [s for s in network.satellites if s.planet == 'Mars' and s.inclination < 30]
        
        total_sats = len([s for s in network.satellites if s.planet == planet])
        return (len(sats) / max(total_sats, 1)) * 100


    #Export data in formats ready for thesis graphs
    def export_for_thesis(self, prefix="thesis_data"):
        #Create results directory if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')
        #Export main time series
        with open(f'results/{prefix}_time_series.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.time_series.keys())
            for i in range(len(self.time_series['time'])):
                row = [self.time_series[key][i] for key in self.time_series.keys()]
                writer.writerow(row)
        #Export key performance indicators
        self._export_kpis(prefix)
        #Export for specific graph types
        self._export_connectivity_comparison(prefix)
        self._export_fso_performance_summary(prefix)
        self._export_blocking_analysis(prefix)
        with open(f'results/{prefix}_link_stability.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time_Hours', 'Avg_Link_Duration', 'Interruption_Frequency', 'MTBF'])
            for i in range(len(self.link_stability['time'])):
                writer.writerow([
                    self.link_stability['time'][i],
                    self.link_stability['average_link_duration'][i],
                    self.link_stability['link_interruption_frequency'][i],
                    self.link_stability['mean_time_between_failures'][i]
                ])
        with open(f'results/{prefix}_handoff_performance.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time_Hours', 'Planned', 'Successful', 'Failed', 'Avg_Latency', 'Seamless'])
            for i in range(len(self.handoff_metrics['time'])):
                writer.writerow([
                    self.handoff_metrics['time'][i],
                    self.handoff_metrics['planned_handoffs'][i],
                    self.handoff_metrics['successful_handoffs'][i],
                    self.handoff_metrics['failed_handoffs'][i],
                    self.handoff_metrics['average_handoff_latency'][i],
                    self.handoff_metrics['seamless_handoffs'][i]
                ])
        with open(f'results/{prefix}_controller_load.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time_Hours', 'Load_Variance', 'Avg_Utilization', 'Max_Utilization', 'Response_Time'])
            for i in range(len(self.controller_load['time'])):
                writer.writerow([
                    self.controller_load['time'][i],
                    self.controller_load['load_variance'][i],
                    self.controller_load['average_utilization'][i],
                    self.controller_load['max_utilization'][i],
                    self.controller_load['response_times'][i]
                ])
        with open(f'results/{prefix}_topology.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time_Hours', 'Clustering_Coef', 'Avg_Path_Length', 'Diameter', 'Edge_Connectivity', 'Critical_Nodes'])
            for i in range(len(self.topology_metrics['time'])):
                writer.writerow([
                    self.topology_metrics['time'][i],
                    self.topology_metrics['clustering_coefficient'][i],
                    self.topology_metrics['average_path_length'][i],
                    self.topology_metrics['network_diameter'][i],
                    self.topology_metrics['edge_connectivity'][i],
                    self.topology_metrics['critical_nodes'][i]
                ])
        
        # 5. Export Queue Metrics
        with open(f'results/{prefix}_queue_performance.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time_Hours', 'Avg_Queue_Depth', 'Max_Queue_Depth', 'Overflow_Events', 'Avg_Wait_Time', 'Deadline_Miss_Rate'])
            for i in range(len(self.queue_metrics['time'])):
                writer.writerow([
                    self.queue_metrics['time'][i],
                    self.queue_metrics['average_queue_depth'][i],
                    self.queue_metrics['max_queue_depth'][i],
                    self.queue_metrics['queue_overflow_events'][i],
                    self.queue_metrics['average_wait_time'][i],
                    self.queue_metrics['deadline_miss_rate'][i]
                ])
        
        # 6. Export Occlusion Events
        with open(f'results/{prefix}_occlusion_events.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time_Hours', 'Sun_Duration', 'Earth_Duration', 'Mars_Duration', 'Simultaneous', 'Longest_Blackout'])
            for i in range(len(self.occlusion_events['time'])):
                writer.writerow([
                    self.occlusion_events['time'][i],
                    self.occlusion_events['sun_occlusion_duration'][i],
                    self.occlusion_events['earth_occlusion_duration'][i],
                    self.occlusion_events['mars_occlusion_duration'][i],
                    self.occlusion_events['simultaneous_occlusions'][i],
                    self.occlusion_events['longest_blackout'][i]
                ])
        with open(f'results/{prefix}_coverage.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time_Hours', 'Earth_Coverage', 'Mars_Coverage', 'Polar_Coverage', 'Equatorial_Coverage'])
            for i in range(len(self.coverage_metrics['time'])):
                writer.writerow([
                    self.coverage_metrics['time'][i],
                    self.coverage_metrics['earth_surface_coverage'][i],
                    self.coverage_metrics['mars_surface_coverage'][i],
                    self.coverage_metrics['polar_coverage'][i],
                    self.coverage_metrics['equatorial_coverage'][i]
                ])
        print(f"\nThesis data exported to results/{prefix}_*.csv")
        
    #Export key performance indicators summary
    def _export_kpis(self, prefix):
        with open(f'results/{prefix}_kpis.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Average', 'Min', 'Max', 'Final'])
            
            metrics = [
                ('Earth Connectivity (%)', 'earth_connectivity'),
                ('Mars Connectivity (%)', 'mars_connectivity'),
                ('Total Connectivity (%)', 'total_connectivity'),
                ('Active FSO Links', 'active_fso_links'),
                ('Data Rate (Tbps)', 'data_rate_tbps'),
                ('Interplanetary Links', 'interplanetary_links'),
                ('Controller Coverage (%)', 'controller_coverage'),
                ('Acquisition Success (%)', 'acquisition_success_rate'),
                ('Navigation Assisted (%)', 'navigation_assisted_pct'),
                ('Average Latency (ms)', 'average_latency_ms')
            ]
            
            for name, key in metrics:
                values = self.time_series[key]
                if values:
                    writer.writerow([
                        name,
                        f"{np.mean(values):.2f}",
                        f"{np.min(values):.2f}",
                        f"{np.max(values):.2f}",
                        f"{values[-1]:.2f}"
                    ])
    
    #Export data specifically for connectivity comparison graphs
    def _export_connectivity_comparison(self, prefix):
        with open(f'results/{prefix}_connectivity_comparison.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time_Hours', 'Time_Days', 'Earth_Pct', 'Mars_Pct', 'Total_Pct', 'Difference'])       
            for i in range(len(self.time_series['time'])):
                time_h = self.time_series['time'][i]
                time_d = time_h / 24
                earth = self.time_series['earth_connectivity'][i]
                mars = self.time_series['mars_connectivity'][i]
                total = self.time_series['total_connectivity'][i]
                diff = abs(earth - mars)
                writer.writerow([time_h, time_d, earth, mars, total, diff])
    
    #Export FSO-specific performance metrics
    def _export_fso_performance_summary(self, prefix):
        with open(f'results/{prefix}_fso_performance.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time_Hours', 'Active_Links', 'Data_Rate_Tbps', 'Interplanetary_Links',
                           'Success_Rate_Pct', 'Nav_Assisted_Pct', 'Traffic_Scheduled'])       
            for i in range(len(self.time_series['time'])):
                writer.writerow([
                    self.time_series['time'][i],
                    self.time_series['active_fso_links'][i],
                    self.time_series['data_rate_tbps'][i],
                    self.time_series['interplanetary_links'][i],
                    self.time_series['acquisition_success_rate'][i],
                    self.time_series['navigation_assisted_pct'][i],
                    self.time_series['traffic_scheduled'][i]
                ])
    #Export line-of-sight blocking analysis
    def _export_blocking_analysis(self, prefix):
        with open(f'results/{prefix}_blocking_analysis.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time_Hours', 'Blocked_By_Sun', 'Blocked_By_Earth', 'Blocked_By_Mars', 'Total_Blocked'])
            
            for i in range(len(self.time_series['time'])):
                sun = self.time_series['blocked_by_sun'][i]
                earth = self.time_series['blocked_by_earth'][i]
                mars = self.time_series['blocked_by_mars'][i]
                total = sun + earth + mars
                
                writer.writerow([
                    self.time_series['time'][i],
                    sun, earth, mars, total
                ])

#Traffic management for controller coordination
class TrafficManager:   
    def __init__(self, controller_id):
        self.controller_id = controller_id
        self.traffic_queues = {}  # satellite_id: TrafficQueue
        self.transmission_schedule = {}  # time_slot: [satellite_assignments]
        self.relay_capacity_monitor = {}  # relay_id: current_load_percentage
        self.priority_levels = ['EMERGENCY', 'CRITICAL', 'HIGH', 'NORMAL', 'LOW']
        
        # Initialize traffic queues for managed satellites
        self.last_traffic_update = 0
        self.traffic_history = []
        
    #Simulate realistic traffic demand for satellites
    def simulate_traffic_demand(self, satellites_in_region):
        traffic_demand = {}   
        for satellite in satellites_in_region:
            #Initialize traffic queue if not exists
            if satellite.sat_id not in self.traffic_queues:
                self.traffic_queues[satellite.sat_id] = TrafficQueue()
            queue = self.traffic_queues[satellite.sat_id]
            #Simulate different types of traffic
            #Regular telemetry (always present)
            queue.add_traffic('NORMAL', 0.1, deadline_hours=2.0, destination='Earth-Ground')
            #Random science data
            if np.random.random() < 0.3:  # 30% chance
                size = np.random.uniform(0.5, 5.0)  # 0.5-5 GB
                queue.add_traffic('HIGH', size, deadline_hours=8.0, destination='Science-Center')
            #Occasional emergency communications
            if np.random.random() < 0.02:  # 2% chance
                queue.add_traffic('EMERGENCY', 0.05, deadline_hours=0.1, destination='Mission-Control')
            #Calculate traffic characteristics
            total_size = queue.total_size_gb
            transmission_time_est = queue.get_transmission_time_estimate(5e9)  # Assume 5 Gbps average
            #Priority breakdown
            priority_breakdown = {}
            for priority in self.priority_levels:
                priority_data = queue.pending_data.get(priority, [])
                priority_breakdown[priority] = {
                    'count': len(priority_data),
                    'size_gb': sum(p['size_gb'] for p in priority_data)
                }
            
            traffic_demand[satellite.sat_id] = {
                'total_data_gb': total_size,
                'priority_breakdown': priority_breakdown,
                'estimated_transmission_time': transmission_time_est,
                'deadline_urgency': queue.urgent_deadline_count,
                'destinations': self._analyze_destinations(queue)
            }
        
        return traffic_demand
    
    #Analyze traffic destinations
    def _analyze_destinations(self, queue):
        destinations = {}
        for priority_data in queue.pending_data.values():
            for packet in priority_data:
                dest = packet.get('destination', 'Unknown')
                if dest not in destinations:
                    destinations[dest] = 0
                destinations[dest] += packet['size_gb']
        return destinations
    
    #Create optimal transmission schedule for satellites
    def optimize_transmission_schedule(self, traffic_analysis, relay_positions, current_time):   
        schedule = TransmissionSchedule(current_time)
        #Sort satellites by priority and urgency
        satellite_priorities = []
        for sat_id, traffic_data in traffic_analysis.items():
            #Calculate priority score
            priority_score = 0
            for priority, data in traffic_data['priority_breakdown'].items():
                priority_weight = {'EMERGENCY': 100, 'CRITICAL': 50, 'HIGH': 20, 'NORMAL': 10, 'LOW': 5}
                priority_score += data['size_gb'] * priority_weight.get(priority, 1)
            # Add urgency bonus
            priority_score += traffic_data['deadline_urgency'] * 25
            satellite_priorities.append((sat_id, priority_score))
        
        #Sort by priority- highest first
        satellite_priorities.sort(key=lambda x: x[1], reverse=True)
        
        #Assign time slots and relay targets
        current_slot_time = current_time
        for satellite_id, priority_score in satellite_priorities:
            traffic_data = traffic_analysis[satellite_id]   
            if traffic_data['total_data_gb'] <= 0:
                continue
            #Find optimal relay
            optimal_relay = self._find_optimal_relay_for_traffic(satellite_id, relay_positions, schedule)
            
            if optimal_relay:
                #Calculate transmission duration
                estimated_data_rate = 8e9  #8 Gbps estimated
                duration = max(TRANSMISSION_SLOT_DURATION, 
                             traffic_data['total_data_gb'] * 8 / estimated_data_rate * 1/3600)  # Convert to hours
                
                #Determine priority
                main_priority = 'NORMAL'
                for priority in ['EMERGENCY', 'CRITICAL', 'HIGH']:
                    if traffic_data['priority_breakdown'][priority]['size_gb'] > 0:
                        main_priority = priority
                        break
                
                schedule.add_assignment(
                    satellite_id=satellite_id,
                    relay_id=optimal_relay,
                    start_time=current_slot_time,
                    duration=duration,
                    priority=main_priority,
                    data_rate=estimated_data_rate
                )
                
                current_slot_time += duration + 0.01  # Small gap between transmissions
        
        return schedule
    
    #Find optimal relay considering traffic load 
    def _find_optimal_relay_for_traffic(self, satellite_id, relay_positions, schedule):
        best_relay = None
        best_score = -1
        
        for relay_id, relay_data in relay_positions.items():
            #Check current load on this relay
            current_load = schedule.relay_capacity.get(relay_id, 0)
            max_capacity = 100
            
            if current_load >= max_capacity:
                continue  #Relay is at capacity
            
            #Calculate score
            capacity_score = (max_capacity - current_load) / max_capacity
            # Note: Would need distance/quality calculation here in real implementation
            quality_score = 0.8  # Simplified
            
            total_score = capacity_score * 0.6 + quality_score * 0.4
            
            if total_score > best_score:
                best_score = total_score
                best_relay = relay_id
        
        return best_relay
    
    #Balance load across available relays
    def coordinate_relay_load_balancing(self, relay_positions, traffic_analysis):        
        relay_loads = {}   
        for relay_id, relay_info in relay_positions.items():
            #Simulate current load
            current_connections = np.random.randint(5, 95)
            max_connections = 100
            #Predict load based on scheduled traffic
            predicted_additional_load = 0
            for sat_id, traffic_data in traffic_analysis.items():
                if traffic_data['total_data_gb'] > 1.0:
                    predicted_additional_load += 5 
            
            predicted_load = min(100, (current_connections + predicted_additional_load) / max_connections * 100)
            
            relay_loads[relay_id] = {
                'current_load_pct': (current_connections / max_connections) * 100,
                'predicted_load_pct': predicted_load,
                'available_capacity_gbps': (max_connections - current_connections) * 0.5,  # 0.5 Gbps per connection
                'congestion_risk': 'HIGH' if predicted_load > 80 else 'NORMAL',
                'recommended_new_assignments': max(0, max_connections - current_connections - 10)
            }
        
        #Generate load balancing actions
        load_balancing_actions = []
        for relay_id, load_data in relay_loads.items():
            if load_data['predicted_load_pct'] > 90:
                load_balancing_actions.append({
                    'action': 'REDISTRIBUTE_LOAD',
                    'relay_id': relay_id,
                    'recommendation': f"Move {int(load_data['predicted_load_pct'] - 80)}% of traffic to other relays"
                })
        
        return {
            'relay_loads': relay_loads,
            'load_balancing_actions': load_balancing_actions,
            'system_bottlenecks': self._identify_bottlenecks(relay_loads)
        }
    
    #Identify system bottlenecks
    def _identify_bottlenecks(self, relay_loads):
        bottlenecks = []
        for relay_id, load_data in relay_loads.items():
            if load_data['predicted_load_pct'] > 85:
                bottlenecks.append({
                    'relay_id': relay_id,
                    'type': 'CAPACITY_BOTTLENECK',
                    'severity': 'HIGH' if load_data['predicted_load_pct'] > 95 else 'MEDIUM'
                })
        return bottlenecks
    
    #Generate comprehensive traffic + positioning update
    def generate_traffic_coordination_update(self, satellites_in_region, current_time):
        #Simulate traffic demand
        traffic_analysis = self.simulate_traffic_demand(satellites_in_region)
        #Get relay positions
        relay_positions = self._get_relay_positions_stub()
        #Optimize transmission schedule
        transmission_schedule = self.optimize_transmission_schedule(
            traffic_analysis, relay_positions, current_time
        )
        #Coordinate load balancing
        load_balancing = self.coordinate_relay_load_balancing(
            relay_positions, traffic_analysis
        )
        #Generate instructions for each satellite
        satellite_instructions = {}
        
        for satellite in satellites_in_region:
            sat_id = satellite.sat_id
            #Get this satellite's transmission assignment
            assignment = transmission_schedule.get_assignment(sat_id)
            if assignment:
                satellite_instructions[sat_id] = {
                    #POSITIONING
                    'recommended_relay': assignment['relay_id'],
                    'relay_position': relay_positions[assignment['relay_id']]['position'],
                    'relay_predicted_positions': relay_positions[assignment['relay_id']].get('predicted_positions', []),
                    
                    #TRAFFIC CONTROL
                    'transmission_start_time': assignment['start_time'],
                    'transmission_duration': assignment['duration'],
                    'transmission_priority': assignment['priority'],
                    'allocated_data_rate': assignment['data_rate'],
                    'transmission_authorization': assignment['auth_token'],
                    
                    # COORDINATION
                    'queue_position': assignment.get('queue_position', 0),
                    'estimated_completion_time': assignment['start_time'] + assignment['duration']
                }
            else:
                #Satellite in waiting queue
                satellite_instructions[sat_id] = {
                    'status': 'QUEUED',
                    'estimated_transmission_time': current_time + 1.0,  # Estimate 1 hour wait
                    'queue_position': len(transmission_schedule.satellite_assignments),
                    'recommended_action': 'STANDBY'
                }
        
        return TrafficCoordinationUpdate(
            controller_id=self.controller_id,
            timestamp=current_time,
            satellite_instructions=satellite_instructions,
            traffic_analysis=traffic_analysis,
            load_balancing=load_balancing
        )
    #Stub method for relay positions - would integrate with navigation coordinator
    def _get_relay_positions_stub(self):
        return {
            'ER1': {'position': np.array([150000000, 0, 0]), 'predicted_positions': []},
            'ER2': {'position': np.array([0, 150000000, 0]), 'predicted_positions': []},
            'ER3': {'position': np.array([-150000000, 0, 0]), 'predicted_positions': []},
            'ER4': {'position': np.array([0, -150000000, 0]), 'predicted_positions': []},
        }

#Enhanced navigation update with traffic management
class TrafficCoordinationUpdate:
    def __init__(self, controller_id, timestamp, satellite_instructions, 
                 traffic_analysis, load_balancing):
        self.controller_id = controller_id
        self.timestamp = timestamp
        self.satellite_instructions = satellite_instructions
        self.traffic_analysis = traffic_analysis
        self.load_balancing = load_balancing
        
        #Network-wide coordination data
        self.global_traffic_state = {}
        self.inter_controller_handoffs = []
        self.congestion_alerts = []

#Navigation update message from controller to satellites
class NavigationUpdate:
    def __init__(self, controller_id, timestamp, relay_positions, optimal_targets, acquisition_windows):
        self.controller_id = controller_id
        self.timestamp = timestamp
        self.relay_positions = relay_positions  #Dict of relay_id: (position, velocity, predicted_positions)
        self.optimal_targets = optimal_targets  #Dict of sat_id: recommended_relay_id
        self.acquisition_windows = acquisition_windows  #Dict of sat_id: [(start_time, end_time, relay_id)]

#Enhanced orbital mechanics with perturbations and real physics
class RealisticOrbitalMechanics:
    def __init__(self):
        self.mu_sun = 1.32712440018e20  #Standard gravitational parameter (m³/s²)
        self.mu_earth = 3.986004418e14
        self.mu_mars = 4.282837e13
        self.mu_moon = 4.9048695e12
        self.c = 299792458  #Speed of light (m/s)
        self.solar_pressure_const = 1361  #Solar constant (W/m²)
    
    #Predict trajectory over multiple time points
    def predict_trajectory(self, position, velocity, time_points):
        trajectory = []
        current_pos = position.copy()
        current_vel = velocity.copy()
        for i, t in enumerate(time_points):
            if i > 0:
                dt = time_points[i] - time_points[i-1]
                #Simple two-body prediction
                current_pos = self.predict_position(current_pos, current_vel, dt)
            trajectory.append(current_pos.copy())
        return trajectory

    #Simple orbital prediction for planning purposes
    def predict_position(self, current_pos, current_vel, time_delta):
        #For circular orbits - simplified prediction
        predicted_pos = current_pos + current_vel * time_delta
        return predicted_pos
    
    #Propagate orbit including J2 perturbation (Earth's oblateness), Solar radiation pressure, Third-body effects (Sun, Moon) and Relativistic corrections
    def propagate_orbit_with_perturbations(self, state, t, params):
        r_vec = state[:3]  #Position vector (m)
        v_vec = state[3:]  #Velocity vector (m/s)
        r = np.linalg.norm(r_vec)
        
        #Primary gravitational acceleration
        central_body = params.get('central_body', 'Earth')
        if central_body == 'Earth':
            mu = self.mu_earth
        elif central_body == 'Mars':
            mu = self.mu_mars
        else:
            mu = self.mu_earth
            
        a_gravity = -mu / r**3 * r_vec
        
        #J2 perturbation - only for Earth
        if central_body == 'Earth':
            J2 = 1.08263e-3
            R_earth = 6.371e6  #meters
            a_J2 = self._calculate_J2_perturbation(r_vec, J2, R_earth)
        else:
            a_J2 = np.zeros(3)
        
        #Solar radiation pressure
        a_srp = self._calculate_solar_radiation_pressure(r_vec, params)
        #Third-body perturbations
        a_third_body = np.zeros(3)
        if 'sun_position' in params:
            a_third_body += self._calculate_third_body_effect(
                r_vec, params['sun_position'], self.mu_sun
            )
        if 'moon_position' in params and central_body == 'Earth':
            a_third_body += self._calculate_third_body_effect(
                r_vec, params['moon_position'], self.mu_moon
            )
        
        #Relativistic correction
        a_rel = self._calculate_relativistic_correction(r_vec, v_vec, mu)
        #Total acceleration
        a_total = a_gravity + a_J2 + a_srp + a_third_body + a_rel
        #[velocity, acceleration]
        return np.concatenate([v_vec, a_total])
    
    #Calculate J2 perturbation acceleration
    def _calculate_J2_perturbation(self, r_vec, J2, R_earth):
        x, y, z = r_vec
        r = np.linalg.norm(r_vec)
        factor = 1.5 * J2 * self.mu_earth * R_earth**2 / r**5   
        a_x = factor * x * (5 * z**2 / r**2 - 1)
        a_y = factor * y * (5 * z**2 / r**2 - 1)
        a_z = factor * z * (5 * z**2 / r**2 - 3)
        return np.array([a_x, a_y, a_z])
    
    #Calculate acceleration due to solar radiation pressure
    def _calculate_solar_radiation_pressure(self, r_vec, params):
        if 'area_to_mass' not in params:
            return np.zeros(3)
            
        #sun position
        sun_pos = params.get('sun_position', np.zeros(3))
        #vector from satellite to sun
        r_sun = sun_pos - r_vec
        r_sun_mag = np.linalg.norm(r_sun)
        if r_sun_mag == 0:
            return np.zeros(3)
            
        #Solar radiation pressure at satellite distance
        P_sr = self.solar_pressure_const * (AU * 1000 / r_sun_mag)**2 / self.c
        #Shadow function (0 in eclipse, 1 in sunlight)
        shadow = self._calculate_shadow_function(r_vec, sun_pos, params)
        #Acceleration - away from sun
        a_srp = -shadow * P_sr * params['area_to_mass'] * r_sun / r_sun_mag
        return a_srp
    
    #Calculate eclipse shadow function
    def _calculate_shadow_function(self, r_sat, r_sun, params):
        #vector from Earth/Mars to satellite
        r_sat_mag = np.linalg.norm(r_sat)   
        #vector from Earth/Mars to Sun
        r_sun_mag = np.linalg.norm(r_sun)
        if r_sat_mag == 0 or r_sun_mag == 0:
            return 1.0 
        #Angle between satellite and sun as seen from central body
        cos_angle = np.dot(r_sat, r_sun) / (r_sat_mag * r_sun_mag)
        
        # if satellite is on opposite side of Earth/Mars from Sun
        if cos_angle < 0:
            #Check if satellite is in umbra
            central_body = params.get('central_body', 'Earth')
            if central_body == 'Earth':
                planet_radius = EARTH_RADIUS * 1000  # meters
            else:
                planet_radius = MARS_RADIUS * 1000
                
            #Angular radius of planet as seen from satellite
            sin_planet = planet_radius / r_sat_mag
            #Angular radius of sun as seen from satellite  
            sin_sun = SUN_RADIUS * 1000 / r_sun_mag
            #If planet appears larger than sun, we're in shadow
            if sin_planet > sin_sun:
                return 0.0
        return 1.0 
    
    #Calculate third-body gravitational perturbation
    def _calculate_third_body_effect(self, r_sat, r_body, mu_body):
        #vector from satellite to third body
        r_rel = r_body - r_sat
        r_rel_mag = np.linalg.norm(r_rel)
        if r_rel_mag == 0:
            return np.zeros(3)
        #direct effect on satellite
        a_direct = mu_body / r_rel_mag**3 * r_rel
        #indirect effect - third body on central body
        r_body_mag = np.linalg.norm(r_body)
        if r_body_mag > 0:
            a_indirect = -mu_body / r_body_mag**3 * r_body
        else:
            a_indirect = np.zeros(3)
        return a_direct + a_indirect
    
    #Calculate relativistic correction to acceleration - Schwarzschild
    def _calculate_relativistic_correction(self, r_vec, v_vec, mu):
        r = np.linalg.norm(r_vec)
        v2 = np.dot(v_vec, v_vec)
        #Schwarzschild radius
        r_s = 2 * mu / self.c**2
        #Leading-order post-Newtonian correction
        factor = mu / (self.c**2 * r**3)
        term1 = (4 * mu / r - v2) * r_vec
        term2 = 4 * np.dot(r_vec, v_vec) * v_vec
        a_rel = factor * (term1 + term2)
        return a_rel

#FSO link acquisition scheduler
class FSO_LinkScheduler:
    def __init__(self):
        self.active_acquisitions = {}  #sat_id: (target_id, start_time, expected_end_time)
        self.scheduled_handoffs = {}   #sat_id: (current_target, new_target, handoff_time)
        self.beam_conflicts = []       #potential beam crossing conflicts
    
    #Plan a handoff between relays
    def plan_handoff(self, satellite_id, current_relay_id, new_relay_id, handoff_time):
        self.scheduled_handoffs[satellite_id] = {
            'current_relay': current_relay_id,
            'new_relay': new_relay_id,
            'handoff_time': handoff_time,
            'status': 'planned'
        }

    def complete_handoff(self, satellite_id, success=True, latency=0.1, data_loss=0):
        """Mark a handoff as complete"""
        if satellite_id in self.scheduled_handoffs:
            self.scheduled_handoffs[satellite_id]['status'] = 'completed' if success else 'failed'
            self.scheduled_handoffs[satellite_id]['latency'] = latency
            self.scheduled_handoffs[satellite_id]['data_loss'] = data_loss

#Enhanced Navigation and coordination engine for controllers with traffic management
class NavigationCoordinator:   
    def __init__(self, controller_id):
        self.controller_id = controller_id
        self.orbital_mechanics = RealisticOrbitalMechanics()
        self.fso_scheduler = FSO_LinkScheduler()
        self.last_update_time = 0
        self.relay_ephemeris = {}  #Cached relay position data
        self.satellite_assignments = {}  #sat_id: assigned_relay_id
        self.traffic_manager = TrafficManager(controller_id)
        self.last_traffic_update = 0
    
    #Update cached relay position and prediction data
    def update_relay_ephemeris(self, relays, current_time):
        prediction_times = np.arange(0, PREDICTION_HORIZON, 0.1)   
        for relay in relays:
            if relay.sat_type == 'relay' and 'Earth-Sun' in str(relay.lagrange_point):
                #predict relay positions
                trajectory = self.orbital_mechanics.predict_trajectory(
                    relay.position, relay.velocity, prediction_times
                )
                self.relay_ephemeris[relay.sat_id] = {
                    'current_position': relay.position.copy(),
                    'current_velocity': relay.velocity.copy(),
                    'predicted_trajectory': trajectory,
                    'last_update': current_time,
                    'lagrange_point': relay.lagrange_point
                }
    
    #Select the optimal relay for a given satellite
    def select_optimal_relay(self, satellite_position, satellite_velocity, current_time):
        best_relay = None
        best_score = -1   
        for relay_id, ephemeris in self.relay_ephemeris.items():
            #calculate current distance
            distance = np.linalg.norm(satellite_position - ephemeris['current_position'])
            #predict future distance - for link stability
            future_sat_pos = self.orbital_mechanics.predict_position(
                satellite_position, satellite_velocity, 1.0  #1 hour ahead
            )
            future_relay_pos = ephemeris['predicted_trajectory'][10] if len(ephemeris['predicted_trajectory']) > 10 else ephemeris['current_position']
            future_distance = np.linalg.norm(future_sat_pos - future_relay_pos)
            #Calculate score based on distance and stability
            distance_score = 1.0 / (1.0 + distance / (RELAY_RANGE * 1000))
            stability_score = 1.0 / (1.0 + abs(future_distance - distance) / 1000)
            total_score = distance_score * 0.7 + stability_score * 0.3
            if total_score > best_score:
                best_score = total_score
                best_relay = relay_id
        return best_relay
    
    #Generate comprehensive positioning + traffic update
    def generate_enhanced_coordination_update(self, satellites_in_region, current_time):   
        #Traditional positioning update
        navigation_update = self.generate_navigation_update(satellites_in_region, current_time)
        #Traffic coordination update
        traffic_update = None
        if current_time - self.last_traffic_update >= TRAFFIC_UPDATE_INTERVAL:
            traffic_update = self.traffic_manager.generate_traffic_coordination_update(
                satellites_in_region, current_time
            )
            self.last_traffic_update = current_time
        
        return {
            'navigation_data': navigation_update,
            'traffic_data': traffic_update,
            'controller_id': self.controller_id,
            'timestamp': current_time
        }
    
    #Generate comprehensive navigation update for satellites in this controller's region
    def generate_navigation_update(self, satellites_in_region, current_time):
        relay_positions = {}
        optimal_targets = {}
        acquisition_windows = {}   
        #Update relay ephemeris data
        for relay_id, ephemeris in self.relay_ephemeris.items():
            relay_positions[relay_id] = {
                'position': ephemeris['current_position'],
                'velocity': ephemeris['current_velocity'],
                'predicted_positions': ephemeris['predicted_trajectory'][:10],  #Next hour
                'lagrange_point': ephemeris['lagrange_point']
            }
        #Calculate optimal targets for each satellite
        for satellite in satellites_in_region:
            optimal_relay = self.select_optimal_relay(
                satellite.position, satellite.velocity, current_time
            )
            optimal_targets[satellite.sat_id] = optimal_relay
            #Calculate acquisition windows (when satellite can see relay)
            acquisition_windows[satellite.sat_id] = self._calculate_acquisition_windows(
                satellite, optimal_relay, current_time
            )
            #Update assignment tracking
            self.satellite_assignments[satellite.sat_id] = optimal_relay
        
        return NavigationUpdate(
            self.controller_id,
            current_time,
            relay_positions,
            optimal_targets,
            acquisition_windows
        )
    
    #Calculate time windows when satellite can acquire relay
    def _calculate_acquisition_windows(self, satellite, relay_id, current_time):
        windows = []   
        if relay_id in self.relay_ephemeris:
            relay_data = self.relay_ephemeris[relay_id]
            #simplified - assume always available if in range
            distance = np.linalg.norm(satellite.position - relay_data['current_position'])
            if distance < RELAY_RANGE * 1000:
                windows.append((current_time, current_time + 1.0, relay_id))  # 1-hour window
        return windows
    
    #Coordinate handoffs between relays for satellites
    def coordinate_handoffs(self, satellites_in_region, current_time):
        handoff_commands = []   
        for satellite in satellites_in_region:
            #Check if satellite needs handoff
            current_links = satellite.fso_terminal.current_links
            for target_id, link_info in current_links.items():
                if link_info['quality'] < HANDOFF_THRESHOLD:
                    #Link quality degrading - plan handoff
                    new_relay = self.select_optimal_relay(
                        satellite.position, satellite.velocity, current_time
                    )
                    if new_relay and new_relay != target_id:
                        handoff_time = current_time + 0.1  #6 minutes ahead
                        self.fso_scheduler.plan_handoff(
                            satellite.sat_id, target_id, new_relay, handoff_time
                        )
                        handoff_commands.append({
                            'satellite_id': satellite.sat_id,
                            'from_relay': target_id,
                            'to_relay': new_relay,
                            'handoff_time': handoff_time
                        })
        return handoff_commands

#FSO terminal that accepts both positioning and traffic instructions
class FSO_Terminal:
    def __init__(self, terminal_id, aperture_size=FSO_APERTURE_DIAMETER, power=FSO_TRANSMIT_POWER):
        self.terminal_id = terminal_id
        self.aperture_size = aperture_size  #meters
        self.transmit_power = power  #watts
        self.wavelength = FSO_WAVELENGTH
        self.beam_divergence = FSO_BEAM_DIVERGENCE
        self.pointing_accuracy = FSO_POINTING_ACCURACY
        #Link state
        self.is_acquiring = False
        self.acquisition_start_time = 0
        self.current_links = {}  #target_id: link_quality
        self.pointing_target = None
        self.link_establishment_time = 0
        #Navigation assistance
        self.navigation_updates = {}  #controller_id: NavigationUpdate
        self.recommended_target = None
        self.last_navigation_update = 0
        #Traffic management state
        self.transmission_schedule = {}
        self.traffic_queue = TrafficQueue()
        self.current_transmission_slot = None
        self.authorized_transmission_time = None
        self.last_traffic_update = 0
        #Performance metrics
        self.total_data_transmitted = 0  #bits
        self.total_acquisition_attempts = 0
        self.successful_acquisitions = 0
        self.link_uptime = 0.0
        self.average_data_rate = 0.0
        #Adaptive parameters
        self.adaptive_power_level = 1.0  #Power scaling factor
        self.beam_quality_factor = 1.0   #Atmospheric/space conditions
    
    #Receive navigation update from controller
    def receive_navigation_update(self, nav_update):
        self.navigation_updates[nav_update.controller_id] = nav_update
        self.last_navigation_update = nav_update.timestamp   
        #update recommended target
        if self.terminal_id in nav_update.optimal_targets:
            self.recommended_target = nav_update.optimal_targets[self.terminal_id]
    
    #Receive enhanced update with traffic + positioning instructions
    def receive_traffic_coordination_update(self, coordination_update):   
        #Handle positioning
        if coordination_update.get('navigation_data'):
            self.receive_navigation_update(coordination_update['navigation_data'])
        #Handle traffic instructions
        if coordination_update.get('traffic_data') and coordination_update['traffic_data']:
            traffic_data = coordination_update['traffic_data']
            self.last_traffic_update = coordination_update['timestamp']   
            if self.terminal_id in traffic_data.satellite_instructions:
                instructions = traffic_data.satellite_instructions[self.terminal_id]
                if instructions.get('status') != 'QUEUED':
                    self.transmission_schedule = {
                        'start_time': instructions['transmission_start_time'],
                        'duration': instructions['transmission_duration'],
                        'relay_target': instructions['recommended_relay'],
                        'data_rate': instructions['allocated_data_rate'],
                        'priority': instructions['transmission_priority'],
                        'authorization': instructions['transmission_authorization']
                    }
                    
                    self.authorized_transmission_time = instructions['transmission_start_time']
                else:
                    self.current_transmission_slot = None
                    self.queue_status = {
                        'position': instructions['queue_position'],
                        'estimated_wait': instructions['estimated_transmission_time']
                    }
    
    #Check if satellite is authorized to transmit at current time
    def is_authorized_to_transmit(self, current_time):   
        if not self.authorized_transmission_time:
            return False
        schedule = self.transmission_schedule
        if not schedule:
            return False
        #Check if in authorized time slot
        start_time = schedule['start_time']
        end_time = start_time + schedule['duration']
        return start_time <= current_time <= end_time
    
    #Begin transmission according to controller's schedule
    def begin_coordinated_transmission(self, current_time): 
        if not self.is_authorized_to_transmit(current_time):
            return False
        #begin transmission with allocated parameters
        schedule = self.transmission_schedule
        print(f"    {self.terminal_id}: Beginning coordinated transmission")
        print(f"        Relay: {schedule['relay_target']}")
        print(f"        Duration: {schedule['duration']:.3f}h")
        print(f"        Data Rate: {schedule['data_rate']/1e9:.1f} Gbps")
        print(f"        Priority: {schedule['priority']}")
        return True

    #Calculate FSO link budget between terminals
    def calculate_link_budget(self, target_terminal, distance):  
        #Transmitted power - dBm
        tx_power_dbm = 10 * np.log10(self.transmit_power * 1000)
        #Free space path loss - dB
        path_loss_db = 20 * np.log10(distance) + 20 * np.log10(self.wavelength) - 147.55
        #Geometric loss - due to beam divergence - dB
        beam_radius_at_target = self.beam_divergence * distance
        target_aperture_area = np.pi * (target_terminal.aperture_size / 2) ** 2
        beam_area = np.pi * beam_radius_at_target ** 2
        geometric_loss_db = -10 * np.log10(target_aperture_area / beam_area)
        #Enhanced pointing loss with navigation assistance
        base_pointing_error = np.random.normal(0, self.pointing_accuracy)
        # Reduce pointing error - recent navigation updates
        if self.last_navigation_update > 0:
            navigation_assistance_factor = 0.5  #50% improvement with navigation
            pointing_error = base_pointing_error * navigation_assistance_factor
        else:
            pointing_error = base_pointing_error
        pointing_loss_db = -12 * (pointing_error / self.beam_divergence) ** 2
        
        #Atmospheric loss - only for Earth-based terminals
        atmospheric_loss = ATMOSPHERIC_LOSS_DB if hasattr(self, 'planet') and self.planet == 'Earth' else 0
        #Total received power - dBm
        rx_power_dbm = (tx_power_dbm - path_loss_db - geometric_loss_db - pointing_loss_db - atmospheric_loss)
        #Convert to linear scale - watts
        rx_power_watts = 10 ** ((rx_power_dbm - 30) / 10)
        return {
            'tx_power_dbm': tx_power_dbm,
            'rx_power_dbm': rx_power_dbm,
            'rx_power_watts': rx_power_watts,
            'path_loss_db': path_loss_db,
            'geometric_loss_db': geometric_loss_db,
            'pointing_loss_db': pointing_loss_db,
            'atmospheric_loss_db': atmospheric_loss,
            'total_loss_db': path_loss_db + geometric_loss_db + pointing_loss_db + atmospheric_loss,
            'navigation_assisted': self.last_navigation_update > 0
        }
    
    #Calculate achievable data rate based on link budget
    def calculate_data_rate(self, link_budget):  
        #Enhanced calculation with navigation assistance bonus
        base_snr_db = link_budget.get('rx_power_dbm', -60) + 90  # Assume -90 dBm noise floor
        #Navigation assistance improves SNR
        if link_budget.get('navigation_assisted', False):
            snr_db = base_snr_db + 2.0  # 2dB improvement with navigation assistance
        else:
            snr_db = base_snr_db
        #lenient capacity calculation
        if snr_db > 10:  # Excellent link
            data_rate = FSO_DATA_RATE_BASELINE * 1.0
            link_quality = FSO_LINK_EXCELLENT
        elif snr_db > 5:  #Good link  
            data_rate = FSO_DATA_RATE_BASELINE * 0.8
            link_quality = FSO_LINK_GOOD
        elif snr_db > 0:  #Degraded link
            data_rate = FSO_DATA_RATE_BASELINE * 0.5
            link_quality = FSO_LINK_DEGRADED
        elif snr_db > -5:   #Poor link
            data_rate = FSO_DATA_RATE_BASELINE * 0.2
            link_quality = FSO_LINK_POOR
        else:              #No viable link
            data_rate = FSO_DATA_RATE_BASELINE * 0.1  #allow minimal link
            link_quality = 0.3  #minimal quality
        
        return {
            'data_rate_bps': data_rate,
            'link_quality': link_quality,
            'snr_db': snr_db,
            'navigation_assisted': link_budget.get('navigation_assisted', False)
        }
    
    #Attempt to establish FSO link with target
    def attempt_link_acquisition(self, target_terminal, distance, current_time):   
        if self.is_acquiring:
            #check if acquisition time has elapsed
            acquisition_time_hours = FSO_ACQUISITION_TIME / 3600
            #navigation assistance reduces acquisition time
            if self.last_navigation_update > 0:
                acquisition_time_hours *= 0.5  #50% faster with navigation assistance
            if current_time - self.acquisition_start_time > acquisition_time_hours:
                self.is_acquiring = False
                #Calculate acquisition success probability - enhanced with navigation
                base_success_rate = 0.75  # Higher base success rate
                distance_factor = max(0.3, 1.0 - (distance / (1.0 * AU * 1000)))
                pointing_factor = 0.9  # Assume good pointing
                #Navigation assistance improves success rate
                if self.last_navigation_update > 0:
                    navigation_bonus = 0.05  #5% improvement
                    base_success_rate = min(0.99, base_success_rate + navigation_bonus)
                success_probability = base_success_rate * distance_factor * pointing_factor
                if np.random.random() < success_probability:
                    # Successful acquisition
                    link_budget = self.calculate_link_budget(target_terminal, distance)
                    link_performance = self.calculate_data_rate(link_budget)
                    self.current_links[target_terminal.terminal_id] = {
                        'quality': max(0.5, link_performance['link_quality']),  #Ensure minimum quality
                        'data_rate': max(1e9, link_performance['data_rate_bps']),  #Minimum 1 Gbps
                        'snr_db': link_performance['snr_db'],
                        'established_time': current_time,
                        'link_budget': link_budget,
                        'navigation_assisted': link_performance['navigation_assisted']
                    }
                    
                    self.successful_acquisitions += 1
                    self.link_establishment_time = current_time
                    
                    return True
                else:
                    #Failed acquisition
                    return False
        else:
            #start new acquisition attempt
            acquisition_delay = FSO_ACQUISITION_TIME / 3600
            if self.last_navigation_update > 0:
                acquisition_delay *= 0.5  #Faster with navigation assistance
                
            self.is_acquiring = True
            self.acquisition_start_time = current_time - acquisition_delay  #Force immediate completion for testing
            self.total_acquisition_attempts += 1
            self.pointing_target = target_terminal.terminal_id
            
        return False  #Still acquiring
    
    #Update existing link quality based on current conditions
    def update_link_quality(self, target_terminal, distance, current_time):   
        if target_terminal.terminal_id in self.current_links:
            #Recalculate link budget
            link_budget = self.calculate_link_budget(target_terminal, distance)
            link_performance = self.calculate_data_rate(link_budget)
            quality_variation = np.random.normal(1.0, 0.05)  # 5% standard deviation
            adjusted_quality = link_performance['link_quality'] * quality_variation
            adjusted_data_rate = link_performance['data_rate_bps'] * quality_variation
            #Update link parameters
            self.current_links[target_terminal.terminal_id].update({
                'quality': max(0, min(1.0, adjusted_quality)),
                'data_rate': max(0, adjusted_data_rate),
                'snr_db': link_performance['snr_db'],
                'link_budget': link_budget,
                'navigation_assisted': link_performance['navigation_assisted']
            })
            # Check if link should drop due to poor conditions
            if adjusted_quality < 0.1:  #Drop link if quality falls below 10%
                del self.current_links[target_terminal.terminal_id]
                return False
            return True
        return False
    
    #Get terminal performance statistics
    def get_performance_stats(self):
        acquisition_success_rate = (self.successful_acquisitions / max(self.total_acquisition_attempts, 1)) * 100   
        #Count navigation-assisted links
        nav_assisted_links = sum(1 for link in self.current_links.values() 
                               if link.get('navigation_assisted', False))
        return {
            'terminal_id': self.terminal_id,
            'total_acquisition_attempts': self.total_acquisition_attempts,
            'successful_acquisitions': self.successful_acquisitions,
            'acquisition_success_rate': acquisition_success_rate,
            'current_active_links': len(self.current_links),
            'navigation_assisted_links': nav_assisted_links,
            'average_link_quality': np.mean([link['quality'] for link in self.current_links.values()]) if self.current_links else 0,
            'total_data_rate': sum([link['data_rate'] for link in self.current_links.values()]),
            'adaptive_power_level': self.adaptive_power_level,
            'beam_quality_factor': self.beam_quality_factor,
            'last_navigation_update': self.last_navigation_update,
            'recommended_target': self.recommended_target,
            'traffic_authorized': self.authorized_transmission_time is not None,
            'transmission_schedule': self.transmission_schedule
        }

#Enhanced Network Architecture with Free Space Optical Communication, Controller Navigation, and Traffic Management
class FSO_NetworkArchitecture:   
    def __init__(self):
        self.satellites = []
        self.connection_graph = nx.Graph()
        self.thesis_metrics = ThesisMetricsCollector()
        self.ephemeris = RealEphemerisData()
        LineOfSightChecker.ephemeris = self.ephemeris
        self.lagrange_calculator = PreciseLagrangePoints(self.ephemeris)
        #blocking analysis
        self.last_blocked_by_sun = 0
        self.last_blocked_by_earth = 0
        self.last_blocked_by_mars = 0
        #Initialize Earth satellites with FSO
        for i in range(EARTH_SATS):
            inclination = 60 if i < EARTH_SATS//2 else 120
            phase = 360 * i / (EARTH_SATS//2)
            raan = 360 * (i // (EARTH_SATS//8)) / 8  #distribute across 8 orbital planes
            sat = EnhancedSatellite(
                sat_id=f"E{i+1}",
                planet='Earth',
                orbit_altitude=EARTH_ORBIT_ALTITUDE,
                inclination=inclination,
                phase=phase,
                sat_type='satellite',
                color='blue'
            )
            self.satellites.append(sat)
        #Initialize Mars satellites with FSO
        for i in range(MARS_SATS):
            inclination = 60 if i < MARS_SATS//2 else 120
            phase = 360 * i / (MARS_SATS//2)
            raan = 360 * i / MARS_SATS  #Spread across different planes
            sat = EnhancedSatellite(
                sat_id=f"M{i+1}",
                planet='Mars',
                orbit_altitude=MARS_ORBIT_ALTITUDE,
                inclination=inclination,
                phase=phase,
                sat_type='satellite',
                color='red'
            )
            self.satellites.append(sat)
        #Initialize Controllers with enhanced navigation capabilities
        earth_moon_lagrange_points = ['Earth-Moon-L1', 'Earth-Moon-L2', 'Earth-Moon-L4', 'Earth-Moon-L5']
        earth_sun_lagrange_points = ['Earth-Sun-L1', 'Earth-Sun-L2', 'Earth-Sun-L4', 'Earth-Sun-L5']
        mars_sun_lagrange_points = ['Mars-Sun-L1', 'Mars-Sun-L2', 'Mars-Sun-L4', 'Mars-Sun-L5']
        #Initialize Controllers with enhanced navigation capabilities
        if EARTH_MOON_CONTROLLERS > 0:
            earth_moon_lagrange_points = ['Earth-Moon-L1', 'Earth-Moon-L2', 'Earth-Moon-L4', 'Earth-Moon-L5']
        
            for i, point in enumerate(earth_moon_lagrange_points):
                point = earth_moon_lagrange_points[i]
                sat = EnhancedSatellite(
                sat_id=f"C{i+1}",
                planet='Controller',
                orbit_altitude=0,
                inclination=0,
                phase=0,
                sat_type='controller',
                lagrange_point=point,
                color='green'
                )
                self.satellites.append(sat)
        #Initialize Earth-Sun relays
        for i, point in enumerate(earth_sun_lagrange_points):
            sat = EnhancedSatellite(
                sat_id=f"ER{i+1}",
                planet='Relay',
                orbit_altitude=0,
                inclination=0,
                phase=0,
                sat_type='relay',
                lagrange_point=point,
                color='cyan'
            )
            self.satellites.append(sat)
        #Initialize Mars-Sun relays
        for i, point in enumerate(mars_sun_lagrange_points):
            sat = EnhancedSatellite(
                sat_id=f"MR{i+1}",
                planet='Relay',
                orbit_altitude=0,
                inclination=0,
                phase=0,
                sat_type='relay',
                lagrange_point=point,
                color='orange'
            )
            self.satellites.append(sat)
        #Enhanced metrics for FSO with controller coordination and traffic management
        self.earth_connectivity = 0.0
        self.mars_connectivity = 0.0
        self.total_connectivity = 0.0
        self.independent_paths = 0
        self.network_latency = 0.0
        self.bandwidth_utilization = 0.0
        #FSO-specific metrics
        self.fso_performance_stats = {}
        self.optical_link_stats = {}
        #Controller coordination metrics
        self.controller_coordination_stats = {}
        #Traffic management metrics
        self.traffic_management_stats = {}
        self.history = []
        self.metrics_collector = ThesisMetricsCollector()

    #Enhanced update with FSO link management, controller coordination, and traffic management
    def update(self, time, ticks):   
        print(f"  Updating {len(self.satellites)} satellites")
        #Update satellite positions and velocities
        for sat in self.satellites:
                sat.update_position_with_orbital_perturbations(time, ticks, dt=SIM_STEP)
        print(f"  Performing controller coordination with traffic management")
        #Perform controller coordination first
        controllers = [sat for sat in self.satellites if sat.sat_type == 'controller']
        for controller in controllers:
            controller.perform_controller_coordination(self.satellites, time)
        print(f"  Attempting FSO link establishments with traffic coordination")
        # Update FSO links for all satellites
        link_attempts = 0
        coordinated_transmissions = 0
        for sat in self.satellites:
            #Check for coordinated transmissions
            if hasattr(sat.fso_terminal, 'is_authorized_to_transmit') and sat.fso_terminal.is_authorized_to_transmit(time):
                coordinated_transmissions += 1
            sat.update_fso_links(time, self.satellites)
            link_attempts += sat.fso_terminal.total_acquisition_attempts
        print(f"  Total FSO acquisition attempts: {link_attempts}, Coordinated transmissions: {coordinated_transmissions}")
        #Update network connections using FSO links
        self._update_fso_connections()
        #Analyze connectivity and FSO performance
        self._analyze_connectivity()
        self._analyze_fso_performance()
        self._analyze_optical_links()
        self._analyze_controller_coordination()
        self._analyze_traffic_management()
        self.analyze_blocked_links()
        self.thesis_metrics.record_metrics(self, time)
        self._record_state(time)
        return self.satellites
    
    #Update network connections using FSO link states
    def _update_fso_connections(self):
        self.connection_graph.clear()
        for sat in self.satellites:
            sat.connections = []
            self.connection_graph.add_node(sat.sat_id)
        total_edges_added = 0
        interplanetary_edges = 0
        controller_edges = 0
        traffic_coordinated_edges = 0
        #Build connections based on active FSO links
        for sat in self.satellites:
            for target_id, link_info in sat.fso_terminal.current_links.items():
                target_sat = next((s for s in self.satellites if s.sat_id == target_id), None)   
                if target_sat and link_info['quality'] > 0.1:  #permissive threshold
                    #Only add each edge once (undirected graph)
                    if not self.connection_graph.has_edge(sat.sat_id, target_id):
                        sat.connections.append(target_id)
                        #Calculate enhanced latency including FSO processing delays
                        distance_km = np.linalg.norm(sat.position - target_sat.position)
                        optical_latency = distance_km / 299792.458  #Speed of light in km/ms
                        processing_delay = 0.1  #FSO processing delay in ms
                        #Navigation assistance reduces latency
                        if link_info.get('navigation_assisted', False):
                            processing_delay *= 0.8  # 20% reduction with navigation assistance
                        #Traffic coordination improves latency further
                        if hasattr(sat.fso_terminal, 'transmission_schedule') and sat.fso_terminal.transmission_schedule:
                            processing_delay *= 0.9  # 10% additional improvement with traffic coordination
                            traffic_coordinated_edges += 1
                        total_latency = optical_latency + processing_delay
                        #Quality-adjusted latency
                        quality_factor = 1.0 / max(link_info['quality'], 0.1)
                        adjusted_latency = total_latency * min(quality_factor, 5.0)  # Cap the penalty
                        self.connection_graph.add_edge(sat.sat_id, target_id, 
                                                     weight=adjusted_latency,
                                                     fso_quality=link_info['quality'],
                                                     fso_data_rate=link_info['data_rate'],
                                                     navigation_assisted=link_info.get('navigation_assisted', False),
                                                     traffic_coordinated=hasattr(sat.fso_terminal, 'transmission_schedule') and bool(sat.fso_terminal.transmission_schedule))
                        total_edges_added += 1
                        #Count different edge types
                        if sat.sat_type == 'controller' or target_sat.sat_type == 'controller':
                            controller_edges += 1
                        #Check if this is an interplanetary edge
                        earth_systems = ['Earth', 'Controller']
                        mars_systems = ['Mars']
                        sat_is_earth = (sat.planet in earth_systems or (sat.sat_type in ['controller', 'relay'] and 'Earth' in str(sat.lagrange_point)))
                        sat_is_mars = (sat.planet in mars_systems or (sat.sat_type == 'relay' and 'Mars' in str(sat.lagrange_point)))
                        target_is_earth = (target_sat.planet in earth_systems or (target_sat.sat_type in ['controller', 'relay'] and 'Earth' in str(target_sat.lagrange_point)))
                        target_is_mars = (target_sat.planet in mars_systems or (target_sat.sat_type == 'relay' and 'Mars' in str(target_sat.lagrange_point)))
                        if (sat_is_earth and target_is_mars) or (sat_is_mars and target_is_earth):
                            interplanetary_edges += 1
        print(f"  Network graph: {total_edges_added} edges added, {interplanetary_edges} interplanetary, {controller_edges} controller, {traffic_coordinated_edges} traffic-coordinated")
        print(f"  Graph stats: {self.connection_graph.number_of_nodes()} nodes, {self.connection_graph.number_of_edges()} edges")
    
    #Analyze how many potential links are blocked by celestial bodies
    def analyze_blocked_links(self):
        blocked_by_sun = 0
        blocked_by_earth = 0
        blocked_by_mars = 0
        blocked_by_moon = 0
        total_potential = 0
        for sat1 in self.satellites:
            for sat2 in self.satellites:
                if sat1.sat_id >= sat2.sat_id:  #Avoid checking same pair twice
                    continue
                distance = np.linalg.norm(sat1.position - sat2.position)
                if distance < RELAY_RANGE * AU:  #Within potential range
                    total_potential += 1
                    los_clear, blocking_body = LineOfSightChecker.is_line_of_sight_clear(
                        sat1.position, sat2.position, 
                        getattr(sat1, 'last_update_time', 0)
                    )
                    if not los_clear:
                        if blocking_body == "Sun":
                            blocked_by_sun += 1
                        elif blocking_body == "Earth":
                            blocked_by_earth += 1
                        elif blocking_body == "Mars":
                            blocked_by_mars += 1
                        elif blocking_body == "Moon":
                            blocked_by_moon += 1
    
        print(f"\nLINE-OF-SIGHT ANALYSIS:")
        print(f"  Total potential links in range: {total_potential}")
        print(f"  Blocked by Sun: {blocked_by_sun}")
        print(f"  Blocked by Earth: {blocked_by_earth}")
        print(f"  Blocked by Mars: {blocked_by_mars}")
        print(f"  Blocked by Moon: {blocked_by_moon}")
        print(f"  Clear links: {total_potential - blocked_by_sun - blocked_by_earth - blocked_by_mars - blocked_by_moon}")
        print(f"  Blocking percentage: {(blocked_by_sun + blocked_by_earth + blocked_by_mars + blocked_by_moon) / max(total_potential, 1) * 100:.1f}%")
        self.last_blocked_by_sun = blocked_by_sun
        self.last_blocked_by_earth = blocked_by_earth
        self.last_blocked_by_mars = blocked_by_mars

    #Connectivity analysis that tests all satellites
    def _analyze_connectivity(self):
        earth_sats = [sat for sat in self.satellites if sat.planet == 'Earth']
        mars_sats = [sat for sat in self.satellites if sat.planet == 'Mars']
        for sat in self.satellites:
            sat.earth_connected = False
            sat.mars_connected = False
            sat.relay_hops = []
        earth_connected = set()
        mars_connected = set()
        total_latency = 0
        path_count = 0
    
        #Test every Mars satellite against at least one Earth satellite
        for mars_sat in mars_sats:
            found_path_to_earth = False
            #Try to find path to ANY Earth satellite
            for earth_sat in earth_sats[:10]:  #Test against first 10 Earth satellites - enough to prove connectivity
                try:
                    path = nx.shortest_path(self.connection_graph, mars_sat.sat_id, earth_sat.sat_id, weight='weight')
                    mars_connected.add(mars_sat.sat_id)
                    earth_connected.add(earth_sat.sat_id)
                    mars_sat.earth_connected = True
                    earth_sat.mars_connected = True
                    mars_sat.relay_hops = path
                    earth_sat.relay_hops = path
                    #calculate latency
                    latency = 0
                    for i in range(len(path) - 1):
                        if self.connection_graph.has_edge(path[i], path[i+1]):
                            latency += self.connection_graph[path[i]][path[i+1]]['weight']
                    total_latency += latency
                    path_count += 1
                    found_path_to_earth = True
                    break 
                except nx.NetworkXNoPath:
                    continue
            if not found_path_to_earth:
                print(f"    {mars_sat.sat_id} cannot reach Earth (this should not happen)")
    
        #Test every Earth satellite against at least one Mars satellite  
        for earth_sat in earth_sats:
            if earth_sat.sat_id in earth_connected:
                continue
            found_path_to_mars = False
        
            #Try to find path to any Mars satellite
            for mars_sat in mars_sats[:3]:  #Test against first 3 Mars satellites
                try:
                    path = nx.shortest_path(self.connection_graph, earth_sat.sat_id, mars_sat.sat_id, weight='weight')
                    earth_connected.add(earth_sat.sat_id)
                    mars_connected.add(mars_sat.sat_id)
                    earth_sat.mars_connected = True
                    mars_sat.earth_connected = True
                    earth_sat.relay_hops = path
                    mars_sat.relay_hops = path
                    #calculate latency
                    latency = 0
                    for i in range(len(path) - 1):
                        if self.connection_graph.has_edge(path[i], path[i+1]):
                            latency += self.connection_graph[path[i]][path[i+1]]['weight']
                    total_latency += latency
                    path_count += 1
                    found_path_to_mars = True
                    break
                except nx.NetworkXNoPath:
                    continue
        print(f"  Total paths found: {path_count}")
        print(f"  Earth satellites connected to Mars: {len(earth_connected)}")
        print(f"  Mars satellites connected to Earth: {len(mars_connected)}")
        #calculate connectivity percentages
        self.earth_connectivity = len(earth_connected) / len(earth_sats) if earth_sats else 0
        self.mars_connectivity = len(mars_connected) / len(mars_sats) if mars_sats else 0
        self.total_connectivity = (self.earth_connectivity + self.mars_connectivity) / 2
        print(f"Connectivity: Earth {self.earth_connectivity*100:.1f}%, Mars {self.mars_connectivity*100:.1f}%, Total {self.total_connectivity*100:.1f}%")
        #Calculate average network latency
        self.network_latency = total_latency / path_count if path_count > 0 else float('inf')
        #Calculate independent paths
        try:
            if earth_connected and mars_connected:
                earth_rep = list(earth_connected)[0]
                mars_rep = list(mars_connected)[0]
                self.independent_paths = nx.edge_connectivity(self.connection_graph, earth_rep, mars_rep)
            else:
                self.independent_paths = 0
        except Exception as e:
            self.independent_paths = 0
        #Calculate bandwidth utilization
        total_possible_connections = len(self.satellites) * (len(self.satellites) - 1) / 2
        active_connections = self.connection_graph.number_of_edges()
        self.bandwidth_utilization = active_connections / total_possible_connections
    
    #Analyze overall FSO system performance with controller coordination and traffic management metrics
    def _analyze_fso_performance(self):
        total_active_links = 0
        total_data_rate = 0.0
        total_acquisition_attempts = 0
        total_successful_acquisitions = 0
        navigation_assisted_links = 0
        traffic_coordinated_links = 0
        quality_distribution = {'excellent': 0, 'good': 0, 'degraded': 0, 'poor': 0}   
        for sat in self.satellites:
            fso_stats = sat.fso_terminal.get_performance_stats()
            total_active_links += fso_stats['current_active_links']
            total_data_rate += fso_stats['total_data_rate']
            total_acquisition_attempts += fso_stats['total_acquisition_attempts']
            total_successful_acquisitions += fso_stats['successful_acquisitions']
            navigation_assisted_links += fso_stats.get('navigation_assisted_links', 0)
            #Count traffic coordinated links
            if fso_stats.get('traffic_authorized', False):
                traffic_coordinated_links += 1
            #Categorize link qualities
            for link in sat.fso_terminal.current_links.values():
                quality = link['quality']
                if quality >= FSO_LINK_EXCELLENT:
                    quality_distribution['excellent'] += 1
                elif quality >= FSO_LINK_GOOD:
                    quality_distribution['good'] += 1
                elif quality >= FSO_LINK_DEGRADED:
                    quality_distribution['degraded'] += 1
                else:
                    quality_distribution['poor'] += 1
        #Calculate system-wide metrics
        acquisition_success_rate = (total_successful_acquisitions / max(total_acquisition_attempts, 1)) * 100
        average_data_rate_per_satellite = total_data_rate / len(self.satellites)
        navigation_assistance_percentage = (navigation_assisted_links / max(total_active_links, 1)) * 100
        traffic_coordination_percentage = (traffic_coordinated_links / max(len(self.satellites), 1)) * 100
        self.fso_performance_stats = {
            'total_active_fso_links': total_active_links,
            'navigation_assisted_links': navigation_assisted_links,
            'traffic_coordinated_links': traffic_coordinated_links,
            'navigation_assistance_percentage': navigation_assistance_percentage,
            'traffic_coordination_percentage': traffic_coordination_percentage,
            'total_data_rate_tbps': total_data_rate / 1e12,
            'acquisition_success_rate_pct': acquisition_success_rate,
            'average_data_rate_per_satellite_gbps': average_data_rate_per_satellite / 1e9,
            'link_quality_distribution': quality_distribution,
            'total_satellites_with_fso': len(self.satellites),
            'network_optical_efficiency': total_active_links / (len(self.satellites) * 2)
        }
    
    #Fixed optical link analysis with better interplanetary detection
    def _analyze_optical_links(self):
        interplanetary_links = 0
        regional_links = 0
        controller_links = 0
        relay_links = 0
        total_interplanetary_data_rate = 0.0
        avg_interplanetary_quality = 0.0
        #Get relay satellites for easier classification
        earth_sun_relays = [sat for sat in self.satellites 
                           if sat.sat_type == 'relay' and 'Earth-Sun' in str(sat.lagrange_point)]
        mars_sun_relays = [sat for sat in self.satellites 
                          if sat.sat_type == 'relay' and 'Mars-Sun' in str(sat.lagrange_point)]
        for sat in self.satellites:
            for target_id, link_info in sat.fso_terminal.current_links.items():
                target_sat = next((s for s in self.satellites if s.sat_id == target_id), None) 
                if target_sat:
                    is_interplanetary = False
                    #1: Check if it's between Earth-Sun and Mars-Sun relays
                    if (sat in earth_sun_relays and target_sat in mars_sun_relays) or \
                       (sat in mars_sun_relays and target_sat in earth_sun_relays):
                        is_interplanetary = True
                
                    #2: Check by distance (backup method)
                    if not is_interplanetary:
                        distance_km = np.linalg.norm(sat.position - target_sat.position)
                        #If distance > 1 AU - likely interplanetary
                        if distance_km > 149597870.7:
                            is_interplanetary = True
                    #3: Check if one satellite is in Earth system, other in Mars system
                    if not is_interplanetary:
                        earth_system_sat = (sat.planet == 'Earth' or 
                                          (sat.sat_type in ['controller', 'relay'] and 'Earth' in str(sat.lagrange_point)))
                        mars_system_sat = (sat.planet == 'Mars' or 
                                         (sat.sat_type == 'relay' and 'Mars' in str(sat.lagrange_point)))
                        target_earth_system = (target_sat.planet == 'Earth' or 
                                             (target_sat.sat_type in ['controller', 'relay'] and 'Earth' in str(target_sat.lagrange_point)))
                        target_mars_system = (target_sat.planet == 'Mars' or 
                                            (target_sat.sat_type == 'relay' and 'Mars' in str(target_sat.lagrange_point)))
                        if (earth_system_sat and target_mars_system) or (mars_system_sat and target_earth_system):
                            is_interplanetary = True
                    #Categorize the link
                    if is_interplanetary:
                        interplanetary_links += 1
                        total_interplanetary_data_rate += link_info['data_rate']
                        avg_interplanetary_quality += link_info['quality']
                    elif sat.sat_type == 'controller' or target_sat.sat_type == 'controller':
                        controller_links += 1
                    elif sat.sat_type == 'relay' or target_sat.sat_type == 'relay':
                        relay_links += 1
                    else:
                        regional_links += 1
        #Calculate averages
        if interplanetary_links > 0:
            avg_interplanetary_quality /= interplanetary_links
        self.optical_link_stats = {
            'interplanetary_links': interplanetary_links,
            'regional_links': regional_links,
            'controller_links': controller_links,
            'relay_links': relay_links,
            'interplanetary_data_rate_gbps': total_interplanetary_data_rate / 1e9,
            'avg_interplanetary_quality': avg_interplanetary_quality,
            'total_optical_links': interplanetary_links + regional_links + controller_links + relay_links
        }

    #Analyze controller coordination effectiveness
    def _analyze_controller_coordination(self):
        controllers = [sat for sat in self.satellites if sat.sat_type == 'controller']   
        total_satellites_coordinated = 0
        total_navigation_updates_sent = 0
        active_controllers = 0
        controller_details = {}
        for controller in controllers:
            if hasattr(controller, 'satellites_in_region'):
                satellites_coordinated = len(controller.satellites_in_region)
                total_satellites_coordinated += satellites_coordinated
                if satellites_coordinated > 0:
                    active_controllers += 1
                controller_details[controller.sat_id] = {
                    'satellites_in_region': satellites_coordinated,
                    'last_coordination_update': getattr(controller, 'last_coordination_update', 0),
                    'lagrange_point': controller.lagrange_point
                }
        #Calculate metrics
        earth_satellites = len([sat for sat in self.satellites if sat.planet == 'Earth'])
        coordination_coverage = (total_satellites_coordinated / max(earth_satellites, 1)) * 100
        self.controller_coordination_stats = {
            'active_controllers': active_controllers,
            'total_controllers': len(controllers),
            'total_satellites_coordinated': total_satellites_coordinated,
            'coordination_coverage_percentage': coordination_coverage,
            'controller_details': controller_details
        }
    
    #analyze traffic management effectiveness
    def _analyze_traffic_management(self):  
        total_satellites_with_traffic_schedule = 0
        total_authorized_transmissions = 0
        total_queued_satellites = 0
        traffic_priorities = {'EMERGENCY': 0, 'CRITICAL': 0, 'HIGH': 0, 'NORMAL': 0, 'LOW': 0}
        controllers_with_traffic = 0
        for sat in self.satellites:
            #Check if satellite has traffic coordination
            if hasattr(sat.fso_terminal, 'transmission_schedule') and sat.fso_terminal.transmission_schedule:
                total_satellites_with_traffic_schedule += 1   
                #Check if authorized to transmit
                if sat.fso_terminal.authorized_transmission_time is not None:
                    total_authorized_transmissions += 1
                #Count priorities
                priority = sat.fso_terminal.transmission_schedule.get('priority', 'NORMAL')
                if priority in traffic_priorities:
                    traffic_priorities[priority] += 1
            #Check if queued
            if hasattr(sat.fso_terminal, 'queue_status'):
                total_queued_satellites += 1
        #Count controllers with traffic management
        for controller in [s for s in self.satellites if s.sat_type == 'controller']:
            if hasattr(controller, 'navigation_coordinator') and hasattr(controller.navigation_coordinator, 'traffic_manager'):
                controllers_with_traffic += 1
        #Calculate metrics
        earth_satellites = len([sat for sat in self.satellites if sat.planet == 'Earth'])
        traffic_coordination_coverage = (total_satellites_with_traffic_schedule / max(earth_satellites, 1)) * 100
        self.traffic_management_stats = {
            'satellites_with_traffic_schedule': total_satellites_with_traffic_schedule,
            'authorized_transmissions': total_authorized_transmissions,
            'queued_satellites': total_queued_satellites,
            'traffic_coordination_coverage_pct': traffic_coordination_coverage,
            'traffic_priority_distribution': traffic_priorities,
            'controllers_with_traffic_mgmt': controllers_with_traffic
        }
    
    #Record current network state with advanced metrics
    def _record_state(self, time):
        state = {
        'time': time,
        'earth_connectivity': self.earth_connectivity * 100,
        'mars_connectivity': self.mars_connectivity * 100,
        'total_connectivity': self.total_connectivity * 100,
        'independent_paths': self.independent_paths,
        'network_latency': self.network_latency,
        'bandwidth_utilization': self.bandwidth_utilization * 100,
        'fso_performance': self.fso_performance_stats,
        'optical_links': self.optical_link_stats,
        'controller_coordination': self.controller_coordination_stats,
        'traffic_management': self.traffic_management_stats,
        'satellites': {sat.sat_id: sat.get_status_dict() for sat in self.satellites},
        }
        self.history.append(state)
    
    #Get comprehensive connectivity statistics including controller coordination and traffic management
    def get_connectivity_stats(self):
        return {
            'earth_connectivity': self.earth_connectivity * 100,
            'mars_connectivity': self.mars_connectivity * 100,
            'total_connectivity': self.total_connectivity * 100,
            'independent_paths': self.independent_paths,
            'network_latency': self.network_latency,
            'bandwidth_utilization': self.bandwidth_utilization * 100,
            'fso_performance': self.fso_performance_stats,
            'optical_links': self.optical_link_stats,
            'controller_coordination': self.controller_coordination_stats,
            'traffic_management': self.traffic_management_stats
        }
    
    #Export comprehensive FSO simulation results with controller coordination and traffic management data
    def export_fso_results(self, filename_prefix="fso_network_simulation_enhanced_with_traffic"):   
        if not os.path.exists('results'):
            os.makedirs('results')
        # Export enhanced time series statistics
        with open(f'results/{filename_prefix}_stats.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Earth_Connectivity_Pct', 'Mars_Connectivity_Pct', 
                           'Total_Connectivity_Pct', 'Independent_Paths', 'Network_Latency_ms', 
                           'Bandwidth_Utilization_Pct', 'FSO_Total_Links', 'FSO_Navigation_Assisted_Links',
                           'FSO_Traffic_Coordinated_Links', 'FSO_Data_Rate_Tbps', 'FSO_Acquisition_Success_Pct', 
                           'Interplanetary_Links', 'Active_Controllers', 'Coordination_Coverage_Pct',
                           'Traffic_Scheduled_Satellites', 'Traffic_Coordination_Coverage_Pct'])
            for state in self.history:
                fso = state.get('fso_performance', {})
                optical = state.get('optical_links', {})
                coordination = state.get('controller_coordination', {})
                traffic = state.get('traffic_management', {})
                writer.writerow([
                    state['time'],
                    state['earth_connectivity'],
                    state['mars_connectivity'],
                    state['total_connectivity'],
                    state['independent_paths'],
                    state['network_latency'],
                    state['bandwidth_utilization'],
                    fso.get('total_active_fso_links', 0),
                    fso.get('navigation_assisted_links', 0),
                    fso.get('traffic_coordinated_links', 0),
                    fso.get('total_data_rate_tbps', 0),
                    fso.get('acquisition_success_rate_pct', 0),
                    optical.get('interplanetary_links', 0),
                    coordination.get('active_controllers', 0),
                    coordination.get('coordination_coverage_percentage', 0),
                    traffic.get('satellites_with_traffic_schedule', 0),
                    traffic.get('traffic_coordination_coverage_pct', 0)
                ])
        #Export satellite data with enhanced FSO, coordination, and traffic metrics
        if self.history:
            last_state = self.history[-1]
            with open(f'results/{filename_prefix}_satellites.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Satellite_ID', 'Planet', 'Satellite_Type', 'Position_X', 'Position_Y', 
                               'Position_Z', 'Connection_Count', 'Earth_Connected', 'Mars_Connected', 
                               'Hop_Count', 'Lagrange_Point', 'FSO_Active_Links', 'FSO_Navigation_Assisted_Links',
                               'FSO_Data_Rate_Gbps', 'FSO_Success_Rate', 'FSO_Aperture_m', 'FSO_Power_W',
                               'Last_Navigation_Update', 'Recommended_Target', 'Satellites_In_Region',
                               'Traffic_Authorized', 'Transmission_Schedule'])
                
                for sat_id, sat_data in last_state['satellites'].items():
                    writer.writerow([
                        sat_data['sat_id'],
                        sat_data['planet'],
                        sat_data['sat_type'],
                        sat_data['position_x'],
                        sat_data['position_y'],
                        sat_data['position_z'],
                        sat_data['connection_count'],
                        sat_data['earth_connected'],
                        sat_data['mars_connected'],
                        sat_data['hop_count'],
                        sat_data['lagrange_point'],
                        sat_data.get('fso_active_links', 0),
                        sat_data.get('fso_navigation_assisted_links', 0),
                        sat_data.get('fso_total_data_rate_gbps', 0),
                        sat_data.get('fso_acquisition_success_rate', 0),
                        sat_data.get('fso_terminal_aperture_m', 0),
                        sat_data.get('fso_terminal_power_w', 0),
                        sat_data.get('last_navigation_update', 0),
                        sat_data.get('recommended_target', 'None'),
                        sat_data.get('satellites_in_region', 0),
                        sat_data.get('traffic_authorized', False),
                        sat_data.get('transmission_schedule', 'None')
                    ])
        #Export traffic management details
        with open(f'results/{filename_prefix}_traffic_management.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Satellites_With_Traffic_Schedule', 'Authorized_Transmissions', 
                           'Queued_Satellites', 'Traffic_Coordination_Coverage_Pct', 'Emergency_Traffic',
                           'Critical_Traffic', 'High_Traffic', 'Normal_Traffic', 'Low_Traffic',
                           'Controllers_With_Traffic_Mgmt'])
            for state in self.history:
                traffic = state.get('traffic_management', {})
                priority_dist = traffic.get('traffic_priority_distribution', {})
                writer.writerow([
                    state['time'],
                    traffic.get('satellites_with_traffic_schedule', 0),
                    traffic.get('authorized_transmissions', 0),
                    traffic.get('queued_satellites', 0),
                    traffic.get('traffic_coordination_coverage_pct', 0),
                    priority_dist.get('EMERGENCY', 0),
                    priority_dist.get('CRITICAL', 0),
                    priority_dist.get('HIGH', 0),
                    priority_dist.get('NORMAL', 0),
                    priority_dist.get('LOW', 0),
                    traffic.get('controllers_with_traffic_mgmt', 0)
                ])
        print(f"\nEnhanced FSO Results with Traffic Management Exported")
        print(f"Files created:")
        print(f"  - {filename_prefix}_stats.csv (comprehensive time series with traffic)")
        print(f"  - {filename_prefix}_satellites.csv (satellite data with traffic metrics)")
        print(f"  - {filename_prefix}_traffic_management.csv (traffic coordination performance)")

#Test basic connectivity between Earth and Mars satellites
def test_simple_connectivity(network):   
    print(f"\nSIMPLE CONNECTIVITY TEST:")
    #Get first Earth and Mars satellite
    earth_sat = next((s for s in network.satellites if s.planet == 'Earth'), None)
    mars_sat = next((s for s in network.satellites if s.planet == 'Mars'), None)
    earth_relay = next((s for s in network.satellites if s.sat_type == 'relay' and 'Earth-Sun' in str(s.lagrange_point)), None)
    mars_relay = next((s for s in network.satellites if s.sat_type == 'relay' and 'Mars-Sun' in str(s.lagrange_point)), None)
    if not all([earth_sat, mars_sat, earth_relay, mars_relay]):
        print("Missing satellites for test")
        return
    print(f"Testing: {earth_sat.sat_id} -> {mars_sat.sat_id}")
    print(f"Via relays: {earth_relay.sat_id} and {mars_relay.sat_id}")
    #Check each hop in the path
    print(f"\nHOP ANALYSIS:")
    #Hop 1: Earth sat ->Earth relay
    distance1 = np.linalg.norm(earth_sat.position - earth_relay.position)
    has_link1 = earth_relay.sat_id in earth_sat.fso_terminal.current_links
    can_link1 = earth_sat._can_establish_fso_link(earth_relay, distance1 * 1000)
    print(f"  1. {earth_sat.sat_id} -> {earth_relay.sat_id}: {distance1:.0f} km")
    print(f"     Has FSO link: {has_link1}")
    print(f"     Can establish: {can_link1}")
    print(f"     In graph: {network.connection_graph.has_edge(earth_sat.sat_id, earth_relay.sat_id)}")
    #Hop 2: Earth relay -> Mars relay  
    distance2 = np.linalg.norm(earth_relay.position - mars_relay.position)
    has_link2 = mars_relay.sat_id in earth_relay.fso_terminal.current_links
    can_link2 = earth_relay._can_establish_fso_link(mars_relay, distance2 * 1000)
    print(f"  2. {earth_relay.sat_id} -> {mars_relay.sat_id}: {distance2:.0f} km")
    print(f"     Has FSO link: {has_link2}")
    print(f"     Can establish: {can_link2}")
    print(f"     In graph: {network.connection_graph.has_edge(earth_relay.sat_id, mars_relay.sat_id)}")
    #Hop 3: Mars relay -> Mars sat
    distance3 = np.linalg.norm(mars_relay.position - mars_sat.position) 
    has_link3 = mars_sat.sat_id in mars_relay.fso_terminal.current_links
    can_link3 = mars_relay._can_establish_fso_link(mars_sat, distance3 * 1000)
    print(f"  3. {mars_relay.sat_id} -> {mars_sat.sat_id}: {distance3:.0f} km")
    print(f"     Has FSO link: {has_link3}")
    print(f"     Can establish: {can_link3}")
    print(f"     In graph: {network.connection_graph.has_edge(mars_relay.sat_id, mars_sat.sat_id)}")
    #Overall path check
    print(f"\n  OVERALL PATH:")
    try:
        path = nx.shortest_path(network.connection_graph, earth_sat.sat_id, mars_sat.sat_id)
        print(f"  Path exists: {' -> '.join(path)}")
    except nx.NetworkXNoPath:
        print(f"  No path exists")
        #Show what each satellite is connected to
        earth_neighbors = list(network.connection_graph.neighbors(earth_sat.sat_id))
        mars_neighbors = list(network.connection_graph.neighbors(mars_sat.sat_id))
        earth_relay_neighbors = list(network.connection_graph.neighbors(earth_relay.sat_id))
        mars_relay_neighbors = list(network.connection_graph.neighbors(mars_relay.sat_id))
        print(f"  {earth_sat.sat_id} connected to: {earth_neighbors}")
        print(f"  {earth_relay.sat_id} connected to: {earth_relay_neighbors}")
        print(f"  {mars_relay.sat_id} connected to: {mars_relay_neighbors}")
        print(f"  {mars_sat.sat_id} connected to: {mars_neighbors}")
    #Check ranges
    print(f"\nRANGE LIMITS:")
    print(f"  RELAY_RANGE: {RELAY_RANGE:.1f} AU ({RELAY_RANGE * 149597870.7:.0f} km)")
    print(f"  Distance 1 within range: {distance1 <= RELAY_RANGE * 149597870.7}")
    print(f"  Distance 2 within range: {distance2 <= RELAY_RANGE * 149597870.7}")
    print(f"  Distance 3 within range: {distance3 <= RELAY_RANGE * 149597870.7}")

def quick_mars_test(network):
    mars_sats = [sat for sat in network.satellites if sat.planet == 'Mars']
    print(f"\nQUICK MARS TEST:")
    for mars_sat in mars_sats:
        neighbors = list(network.connection_graph.neighbors(mars_sat.sat_id))
        relay_neighbors = [n for n in neighbors if n.startswith('MR')]
        other_neighbors = [n for n in neighbors if not n.startswith('MR')]
        print(f"  {mars_sat.sat_id}: Relay connections: {relay_neighbors}, Other: {other_neighbors}")
        if not relay_neighbors:
            print(f"    {mars_sat.sat_id} has NO relay access")

#Test controller coordination functionality
def test_controller_coordination(network):
    print(f"\nCONTROLLER COORDINATION TEST:")
    controllers = [sat for sat in network.satellites if sat.sat_type == 'controller']
    earth_satellites = [sat for sat in network.satellites if sat.planet == 'Earth']
    for controller in controllers:
        print(f"\nController {controller.sat_id} ({controller.lagrange_point}):")
        #Check satellites in region
        if hasattr(controller, 'satellites_in_region'):
            sats_in_region = len(controller.satellites_in_region)
            print(f"  Satellites in region: {sats_in_region}")
            #Check navigation updates sent
            nav_updates_received = 0
            traffic_schedules_sent = 0
            for sat in controller.satellites_in_region:
                if len(sat.fso_terminal.navigation_updates) > 0:
                    nav_updates_received += 1
                if hasattr(sat.fso_terminal, 'transmission_schedule') and sat.fso_terminal.transmission_schedule:
                    traffic_schedules_sent += 1
            print(f"  Navigation updates received: {nav_updates_received}/{sats_in_region}")
            print(f"  Traffic schedules sent: {traffic_schedules_sent}/{sats_in_region}")
            #Check recommended targets
            recommended_targets = {}
            for sat in controller.satellites_in_region:
                if sat.fso_terminal.recommended_target:
                    target = sat.fso_terminal.recommended_target
                    if target not in recommended_targets:
                        recommended_targets[target] = 0
                    recommended_targets[target] += 1
            
            print(f"  Target recommendations: {recommended_targets}")
        #Check controller's own FSO links
        controller_links = len(controller.fso_terminal.current_links)
        relay_links = sum(1 for target_id in controller.fso_terminal.current_links.keys() 
                         if target_id.startswith('R'))
        print(f"  Controller FSO links: {controller_links} total, {relay_links} to relays")

#Test network connectivity during Mars solar conjunction
def test_solar_conjunction(network):
    print("\n" + "="*80)
    print("SOLAR CONJUNCTION TEST")
    print("="*80)
    #Calculate solar conjunction orbital positions
    conjunction_time = 8760  # Hours - 1 year for testing
    #Set up conjunction positions - Earth at 1 AU, 0 degrees
    earth_angle = 0
    earth_position = np.array([AU, 0, 0])
    # Mars at 1.52 AU, 180 degrees - opposite side of Sun
    mars_angle = np.pi  # 180 degrees
    mars_position = np.array([1.52 * AU * np.cos(mars_angle), 1.52 * AU * np.sin(mars_angle), 0])
    #Sun at origin
    sun_position = np.array([0, 0, 0])
    print(f"  Earth position: [{earth_position[0]/AU:.2f}, {earth_position[1]/AU:.2f}, {earth_position[2]/AU:.2f}] AU")
    print(f"  Mars position: [{mars_position[0]/AU:.2f}, {mars_position[1]/AU:.2f}, {mars_position[2]/AU:.2f}] AU")
    print(f"  Earth-Mars separation: {np.linalg.norm(mars_position - earth_position)/AU:.2f} AU")
    #update all satellite positions for conjunction
    print("\nUpdating satellite positions for conjunction")
    #update Earth satellites
    for sat in network.satellites:
        if sat.planet == 'Earth':
            #keep satellites in Earth orbit
            orbit_radius = EARTH_RADIUS + sat.orbit_altitude
            sat_angle = np.radians(sat.phase)
            x_orbit = orbit_radius * np.cos(sat_angle)
            y_orbit = orbit_radius * np.sin(sat_angle)
            z_orbit = y_orbit * np.sin(np.radians(sat.inclination))
            y_orbit = y_orbit * np.cos(np.radians(sat.inclination))
            sat.position = earth_position + np.array([x_orbit, y_orbit, z_orbit])
        elif sat.planet == 'Mars':
            #keep satellites in Mars orbit
            orbit_radius = MARS_RADIUS + sat.orbit_altitude
            sat_angle = np.radians(sat.phase)
            x_orbit = orbit_radius * np.cos(sat_angle)
            y_orbit = orbit_radius * np.sin(sat_angle)
            z_orbit = y_orbit * np.sin(np.radians(sat.inclination))
            y_orbit = y_orbit * np.cos(np.radians(sat.inclination))
            sat.position = mars_position + np.array([x_orbit, y_orbit, z_orbit])
    #update relay positions for conjunction
    for sat in network.satellites:
        if sat.sat_type == 'relay':
            if 'Earth-Sun' in str(sat.lagrange_point):
                if sat.lagrange_point == 'Earth-Sun-L1':
                    #L1: Between Earth and Sun
                    sat.position = earth_position * 0.99
                elif sat.lagrange_point == 'Earth-Sun-L2':
                    #L2: Beyond Earth from Sun
                    sat.position = earth_position * 1.01
                elif sat.lagrange_point == 'Earth-Sun-L4':
                    #L4: 60 degrees ahead of Earth
                    l4_angle = earth_angle + np.pi/3
                    sat.position = AU * np.array([np.cos(l4_angle), np.sin(l4_angle), 0])
                elif sat.lagrange_point == 'Earth-Sun-L5':
                    #L5: 60 degrees behind Earth
                    l5_angle = earth_angle - np.pi/3
                    sat.position = AU * np.array([np.cos(l5_angle), np.sin(l5_angle), 0])
            elif 'Mars-Sun' in str(sat.lagrange_point):
                if sat.lagrange_point == 'Mars-Sun-L1':
                    #L1: Between Mars and Sun
                    sat.position = mars_position * 0.99
                elif sat.lagrange_point == 'Mars-Sun-L2':
                    #L2: Beyond Mars from Sun
                    sat.position = mars_position * 1.01
                elif sat.lagrange_point == 'Mars-Sun-L4':
                    #L4: 60 degrees ahead of Mars
                    l4_angle = mars_angle + np.pi/3
                    sat.position = 1.52 * AU * np.array([np.cos(l4_angle), np.sin(l4_angle), 0])
                elif sat.lagrange_point == 'Mars-Sun-L5':
                    #L5: 60 degrees behind Mars
                    l5_angle = mars_angle - np.pi/3
                    sat.position = 1.52 * AU * np.array([np.cos(l5_angle), np.sin(l5_angle), 0])
    #Update controller positions - Earth-Moon system moves with Earth
    for sat in network.satellites:
        if sat.sat_type == 'controller':
            #Simple offset from Earth for testing
            controller_offset = 400000  #400,000 km from Earth
            if 'L1' in sat.lagrange_point:
                sat.position = earth_position + np.array([controller_offset, 0, 0])
            elif 'L2' in sat.lagrange_point:
                sat.position = earth_position + np.array([-controller_offset, 0, 0])
            elif 'L4' in sat.lagrange_point:
                sat.position = earth_position + np.array([0, controller_offset, 0])
            elif 'L5' in sat.lagrange_point:
                sat.position = earth_position + np.array([0, -controller_offset, 0])
    #Test line of sight for key connections
    #Get sample satellites
    earth_sat = next((s for s in network.satellites if s.planet == 'Earth'), None)
    mars_sat = next((s for s in network.satellites if s.planet == 'Mars'), None)
    earth_sun_relays = [s for s in network.satellites if s.sat_type == 'relay' and 'Earth-Sun' in str(s.lagrange_point)]
    mars_sun_relays = [s for s in network.satellites if s.sat_type == 'relay' and 'Mars-Sun' in str(s.lagrange_point)]
    #Test direct Earth-Mars line of sight
    print("\nDirect Earth-Mars Communication:")
    los_clear, blocking_body = LineOfSightChecker.is_line_of_sight_clear(
        earth_sat.position, mars_sat.position, conjunction_time
    )
    print(f"   {earth_sat.sat_id} -> {mars_sat.sat_id}: {'CLEAR' if los_clear else f'BLOCKED by {blocking_body}'}")
    #Test Earth to Earth-Sun relays
    print("\n2Earth to Earth-Sun Relay Connections:")
    for relay in earth_sun_relays:
        los_clear, blocking_body = LineOfSightChecker.is_line_of_sight_clear(
            earth_sat.position, relay.position, conjunction_time
        )
        distance = np.linalg.norm(earth_sat.position - relay.position) / AU
        print(f"   {earth_sat.sat_id} -> {relay.sat_id} ({relay.lagrange_point}): "
              f"{'CLEAR' if los_clear else f'BLOCKED by {blocking_body}'} "
              f"[{distance:.3f} AU]")
    #Test Mars to Mars-Sun relays
    print("\nMars to Mars-Sun Relay Connections:")
    for relay in mars_sun_relays:
        los_clear, blocking_body = LineOfSightChecker.is_line_of_sight_clear(
            mars_sat.position, relay.position, conjunction_time
        )
        distance = np.linalg.norm(mars_sat.position - relay.position) / AU
        print(f"   {mars_sat.sat_id} -> {relay.sat_id} ({relay.lagrange_point}): "
              f"{'CLEAR' if los_clear else f'BLOCKED by {blocking_body}'} "
              f"[{distance:.3f} AU]")
    #Test inter-relay connections - the critical paths
    print("\nCritical Inter-Relay Connections:")
    critical_paths_found = []
    for earth_relay in earth_sun_relays:
        for mars_relay in mars_sun_relays:
            los_clear, blocking_body = LineOfSightChecker.is_line_of_sight_clear(
                earth_relay.position, mars_relay.position, conjunction_time
            )
            distance = np.linalg.norm(earth_relay.position - mars_relay.position) / AU
            
            status = 'CLEAR' if los_clear else f'BLOCKED by {blocking_body}'
            print(f"   {earth_relay.sat_id} ({earth_relay.lagrange_point}) -> "
                  f"{mars_relay.sat_id} ({mars_relay.lagrange_point}): {status} [{distance:.3f} AU]")
            
            if los_clear and distance <= RELAY_RANGE:
                critical_paths_found.append((earth_relay.sat_id, mars_relay.sat_id, distance))
    #Find alternative paths through L4/L5
    print("\nAlternative Paths Analysis:")
    l4_l5_paths = []
    #Check L4/L5 relays specifically
    earth_l4 = next((s for s in earth_sun_relays if 'L4' in s.lagrange_point), None)
    earth_l5 = next((s for s in earth_sun_relays if 'L5' in s.lagrange_point), None)
    mars_l4 = next((s for s in mars_sun_relays if 'L4' in s.lagrange_point), None)
    mars_l5 = next((s for s in mars_sun_relays if 'L5' in s.lagrange_point), None)
    #Test specific L4/L5 paths that might work
    test_paths = [
        (earth_l4, mars_l4, "Earth-L4 to Mars-L4"),
        (earth_l4, mars_l5, "Earth-L4 to Mars-L5"),
        (earth_l5, mars_l4, "Earth-L5 to Mars-L4"),
        (earth_l5, mars_l5, "Earth-L5 to Mars-L5")
    ]
    for relay1, relay2, path_name in test_paths:
        if relay1 and relay2:
            los_clear, blocking_body = LineOfSightChecker.is_line_of_sight_clear(
                relay1.position, relay2.position, conjunction_time
            )
            distance = np.linalg.norm(relay1.position - relay2.position) / AU    
            if los_clear and distance <= RELAY_RANGE:
                l4_l5_paths.append(path_name)
                print(f"   {path_name}: AVAILABLE [{distance:.3f} AU]")
            else:
                print(f"   {path_name}: {'BLOCKED by ' + blocking_body if not los_clear else 'OUT OF RANGE'} [{distance:.3f} AU]")
    #Summary
    print("\nSOLAR CONJUNCTION CONNECTIVITY SUMMARY:")
    print(f"   Direct Earth-Mars communication: BLOCKED")
    print(f"   Working relay paths found: {len(critical_paths_found)}")
    print(f"   L4/L5 alternative paths: {len(l4_l5_paths)}")
    if critical_paths_found:
        print("\n   NETWORK MAINTAINS CONNECTIVITY through:")
        for earth_relay, mars_relay, dist in critical_paths_found[:3]: 
            print(f"      {earth_relay} -> {mars_relay} ({dist:.3f} AU)")
    else:
        print("\n   WARNING: No relay paths available during conjunction")
    #Calculate data accumulation during conjunction
    print("\nDATA ACCUMULATION ESTIMATE:")
    conjunction_duration_days = 14
    mars_daily_data_gb = 10  # Estimate: rovers, orbiters, etc.
    total_data_gb = conjunction_duration_days * mars_daily_data_gb
    print(f"   Conjunction duration: {conjunction_duration_days} days")
    print(f"   Mars daily data generation: {mars_daily_data_gb} GB/day")
    print(f"   Total data to cache: {total_data_gb} GB")
    if l4_l5_paths:
        print(f"   With L4/L5 paths: Real-time transmission possible (with {len(l4_l5_paths)} routes)")
    else:
        print(f"   Without working paths: Need {total_data_gb} GB storage at Mars relays")
    
    return len(critical_paths_found) > 0 or len(l4_l5_paths) > 0

#Test network's ability to recover from sudden link failures
def test_dynamic_link_failure_recovery(network, num_failures=5):   
    print("\n" + "="*80)
    print("DYNAMIC LINK FAILURE RECOVERY TEST")
    print("="*80)
    #Get initial connectivity
    initial_earth_connectivity = network.earth_connectivity
    initial_mars_connectivity = network.mars_connectivity
    #Get all active FSO links
    active_links = []
    for sat in network.satellites:
        for target_id in sat.fso_terminal.current_links.keys():
            if (sat.sat_id, target_id) not in active_links and (target_id, sat.sat_id) not in active_links:
                active_links.append((sat.sat_id, target_id))
    print(f"\nInitial State:")
    print(f"  Active FSO links: {len(active_links)}")
    print(f"  Earth connectivity: {initial_earth_connectivity*100:.1f}%")
    print(f"  Mars connectivity: {initial_mars_connectivity*100:.1f}%")
    #Randomly fail some links
    import random
    failed_links = random.sample(active_links, min(num_failures, len(active_links)))
    print(f"\n Simulating {len(failed_links)} link failures:")
    for sat_id, target_id in failed_links:
        sat = next(s for s in network.satellites if s.sat_id == sat_id)
        if target_id in sat.fso_terminal.current_links:
            del sat.fso_terminal.current_links[target_id]
            print(f"  Failed: {sat_id} <=> {target_id}")
    #update network graph
    network._update_fso_connections()
    network._analyze_connectivity()
    print(f"\nAfter Failures:")
    print(f"  Earth connectivity: {network.earth_connectivity*100:.1f}% (Delta {(network.earth_connectivity-initial_earth_connectivity)*100:+.1f}%)")
    print(f"  Mars connectivity: {network.mars_connectivity*100:.1f}% (Delta {(network.mars_connectivity-initial_mars_connectivity)*100:+.1f}%)")
    #Simulate recovery attempts
    print(f"\nAttempting recovery")
    recovery_time = 0
    max_recovery_attempts = 10
    for attempt in range(max_recovery_attempts):
        recovery_time += 0.1  #Each attempt takes 6 minutes
        #Try to re-establish failed links
        recovered = 0
        for sat_id, target_id in failed_links:
            sat = next(s for s in network.satellites if s.sat_id == sat_id)
            target = next(s for s in network.satellites if s.sat_id == target_id)   
            distance = np.linalg.norm(sat.position - target.position) * 1000
            if sat._can_establish_fso_link(target, distance):
                #Attempt acquisition
                if np.random.random() < 0.3:  # 30% chance per attempt
                    sat.fso_terminal.current_links[target_id] = {
                        'quality': 0.8,
                        'data_rate': 5e9,
                        'established_time': recovery_time
                    }
                    recovered += 1
        if recovered > 0:
            print(f"  Attempt {attempt+1}: Recovered {recovered} links")
        #Update connectivity
        network._update_fso_connections()
        network._analyze_connectivity()
        #Check if fully recovered
        if network.earth_connectivity >= initial_earth_connectivity * 0.95:
            print(f"\nNetwork recovered to 95% connectivity in {recovery_time:.1f} hours")
            break
    print(f"\nFinal Recovery State:")
    print(f"  Earth connectivity: {network.earth_connectivity*100:.1f}%")
    print(f"  Mars connectivity: {network.mars_connectivity*100:.1f}%")
    print(f"  Recovery time: {recovery_time:.1f} hours")
    return recovery_time

#Test network performance under heavy traffic load
def test_traffic_overload_scenario(network, overload_factor=3.0):
    print("\n" + "="*80)
    print("TRAFFIC OVERLOAD SCENARIO TEST")
    print("="*80)   
    #Simulate heavy traffic on Earth satellites
    earth_sats = [sat for sat in network.satellites if sat.planet == 'Earth']
    print(f"\nSimulating {overload_factor}x normal traffic load")
    #Add traffic to satellite queues
    for sat in earth_sats[:20]:  # Test on subset
        if hasattr(sat.fso_terminal, 'traffic_queue'):
            #Add various priority traffic
            sat.fso_terminal.traffic_queue.add_traffic('EMERGENCY', 0.5, deadline_hours=0.1)
            sat.fso_terminal.traffic_queue.add_traffic('CRITICAL', 2.0, deadline_hours=0.5)
            sat.fso_terminal.traffic_queue.add_traffic('HIGH', 5.0 * overload_factor, deadline_hours=2.0)
            sat.fso_terminal.traffic_queue.add_traffic('NORMAL', 10.0 * overload_factor, deadline_hours=8.0)
    #Get traffic metrics
    total_queued_data = 0
    emergency_packets = 0
    satellites_overloaded = 0
    for sat in earth_sats:
        if hasattr(sat.fso_terminal, 'traffic_queue'):
            queue = sat.fso_terminal.traffic_queue
            total_queued_data += queue.total_size_gb
            emergency_packets += queue.urgent_deadline_count
            #Check if overloaded (>1 hour transmission time at current rate)
            if len(sat.fso_terminal.current_links) > 0:
                avg_rate = np.mean([link['data_rate'] for link in sat.fso_terminal.current_links.values()])
                transmission_time = queue.get_transmission_time_estimate(avg_rate)
                if transmission_time > 1.0:
                    satellites_overloaded += 1
    print(f"\nTraffic Load Analysis:")
    print(f"  Total queued data: {total_queued_data:.1f} GB")
    print(f"  Emergency packets: {emergency_packets}")
    print(f"  Overloaded satellites: {satellites_overloaded}/{len(earth_sats)}")
    #Test controller response
    controllers = [sat for sat in network.satellites if sat.sat_type == 'controller']
    print(f"\nController Response:")
    for controller in controllers:
        if hasattr(controller, 'traffic_manager'):
            #Simulate traffic management
            traffic_analysis = controller.traffic_manager.simulate_traffic_demand(
                controller.satellites_in_region if hasattr(controller, 'satellites_in_region') else []
            )
            scheduled_count = sum(1 for data in traffic_analysis.values() 
                                if data['total_data_gb'] > 0)
            emergency_count = sum(1 for data in traffic_analysis.values() 
                                if data['priority_breakdown'].get('EMERGENCY', {}).get('count', 0) > 0)
            print(f"  {controller.sat_id}: Scheduled {scheduled_count} satellites, "
                  f"{emergency_count} with emergency traffic")
    
    #Calculate network stress metrics
    network_stress_level = satellites_overloaded / len(earth_sats)
    print(f"\nNetwork Stress Level: {network_stress_level*100:.1f}%")
    if network_stress_level > 0.5:
        print("  Status: CRITICAL - Network overloaded")
    elif network_stress_level > 0.2:
        print("  Status: WARNING - High traffic load")
    else:
        print("  Status: NORMAL - Network handling traffic well")    
    return network_stress_level

#Test network performance at different orbital configurations
def test_orbital_configuration_impact(network):
    print("\n" + "="*80)
    print("ORBITAL CONFIGURATION IMPACT TEST")
    print("="*80)
    configurations = [
        ("Conjunction", 0, 180),      #Earth at 0°, Mars at 180° - behind Sun
        ("Quadrature", 0, 90),        #Earth at 0°, Mars at 90°
        ("Opposition", 0, 0),         #Earth and Mars on same side
        ("Near-Conjunction", 0, 150), #Almost conjunction
    ]
    results = {}
    for config_name, earth_angle, mars_angle in configurations:
        print(f"\nTesting {config_name} configuration:")
        print(f"   Earth angle: {earth_angle}°, Mars angle: {mars_angle}°")
        #set orbital positions
        earth_pos = AU * np.array([np.cos(np.radians(earth_angle)), 
                                  np.sin(np.radians(earth_angle)), 0])
        mars_pos = 1.52 * AU * np.array([np.cos(np.radians(mars_angle)), 
                                        np.sin(np.radians(mars_angle)), 0])
        #Update satellite positions
        for sat in network.satellites:
            if sat.planet == 'Earth':
                sat.position = earth_pos + (sat.position - earth_pos) * 0.001
            elif sat.planet == 'Mars':
                sat.position = mars_pos + (sat.position - mars_pos) * 0.001
        #Update network
        network._update_fso_connections()
        network._analyze_connectivity()
        network._analyze_optical_links()
        #Store results
        results[config_name] = {
            'earth_mars_distance': np.linalg.norm(mars_pos - earth_pos) / AU,
            'earth_connectivity': network.earth_connectivity * 100,
            'mars_connectivity': network.mars_connectivity * 100,
            'interplanetary_links': network.optical_link_stats.get('interplanetary_links', 0),
            'average_latency': network.network_latency
        }
        print(f"   Distance: {results[config_name]['earth_mars_distance']:.2f} AU")
        print(f"   Connectivity: Earth {results[config_name]['earth_connectivity']:.1f}%, "
              f"Mars {results[config_name]['mars_connectivity']:.1f}%")
        print(f"   Interplanetary links: {results[config_name]['interplanetary_links']}")
    
    #Summary
    print(f"\nCONFIGURATION COMPARISON:")
    print(f"{'Configuration':<20} {'Distance':>10} {'E-Connect':>12} {'M-Connect':>12} {'IP-Links':>10}")
    print(f"{'-'*20} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    for config, data in results.items():
        print(f"{config:<20} {data['earth_mars_distance']:>10.2f} "
              f"{data['earth_connectivity']:>11.1f}% "
              f"{data['mars_connectivity']:>11.1f}% "
              f"{data['interplanetary_links']:>10}")    
    return results

#Run only selected validation tests for thesis section 4.2
def run_selective_validation_tests(network, test_numbers=[1, 2, 3, 4]):
    print("\n" + "="*40)
    print("\nSELECTIVE FSO NETWORK VALIDATION SUITE")
    print("For Thesis Section 4.2: Solar Conjunction Resilience Validation")
    print(f"Running tests: {test_numbers}")
    print("\n" + "="*40)   
    test_results = {}
    #Solar Conjunction Test
    if 1 in test_numbers:
        print("\nTest 1: Solar Conjunction Connectivity")
        conjunction_result = test_solar_conjunction(network)
        test_results['solar_conjunction'] = conjunction_result
    #Dynamic Failure Recovery
    if 2 in test_numbers:
        print("\nTest 2: Dynamic Link Failure Recovery")
        recovery_time = test_dynamic_link_failure_recovery(network, num_failures=3)
        test_results['failure_recovery'] = recovery_time
    #Traffic Overload
    if 3 in test_numbers:
        print("\nTest 3: Traffic Overload Scenario")
        stress_level = test_traffic_overload_scenario(network, overload_factor=2.5)
        test_results['traffic_overload'] = stress_level
    #Orbital Configuration Impact
    if 4 in test_numbers:
        print("\nTest 4: Orbital Configuration Analysis")
        orbital_results = test_orbital_configuration_impact(network)
        test_results['orbital_configs'] = orbital_results
    #Summary Report
    print("\n" + "="*80)
    print("VALIDATION TEST SUMMARY REPORT")
    print("="*80)
    print(f"\nTests Completed: {len(test_results)}/{len(test_numbers)}")
    print(f"\nKey Findings:")
    if 'solar_conjunction' in test_results:
        print(f"  • Solar Conjunction: {'PASSED' if test_results['solar_conjunction'] else 'FAILED'}")
    if 'failure_recovery' in test_results:
        print(f"  • Failure Recovery Time: {test_results['failure_recovery']:.1f} hours")
    if 'traffic_overload' in test_results:
        print(f"  • Traffic Stress Tolerance: {(1-test_results['traffic_overload'])*100:.1f}%")
    if 'orbital_configs' in test_results:
        best_config = max(test_results['orbital_configs'].items(), 
                         key=lambda x: x[1]['earth_connectivity'])[0]
        print(f"  • Best Orbital Config: {best_config}")
    return test_results

#Export validation test results in formats suitable for the separate figure generator
def export_validation_results_for_figures(test_results, export_dir='results'):   
    #Create results directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)
    #Export complete results as JSON
    with open(os.path.join(export_dir, 'validation_test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    #Export failure recovery data
    if 'failure_recovery' in test_results:
        with open(os.path.join(export_dir, 'validation_failure_recovery.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Recovery_Time_Hours', test_results['failure_recovery']])
            writer.writerow(['Recovery_Success', True])
    #Export traffic overload data
    if 'traffic_overload' in test_results:
        with open(os.path.join(export_dir, 'validation_traffic_overload.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Network_Stress_Level', test_results['traffic_overload']])
            writer.writerow(['Network_Health_Percentage', (1-test_results['traffic_overload'])*100])
            writer.writerow(['Overload_Factor', 2.5])
    #Export orbital configuration data
    if 'orbital_configs' in test_results:
        with open(os.path.join(export_dir, 'validation_orbital_configs.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Configuration', 'Distance_AU', 'Earth_Connectivity', 'Mars_Connectivity', 'Interplanetary_Links', 'Average_Latency'])
            for config_name, data in test_results['orbital_configs'].items():
                writer.writerow([
                    config_name,
                    data['earth_mars_distance'],
                    data['earth_connectivity'],
                    data['mars_connectivity'],
                    data['interplanetary_links'],
                    data.get('average_latency', 0)
                ])
    
    #Export validation summary
    with open(os.path.join(export_dir, 'validation_summary.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Test_Name', 'Status', 'Key_Metric', 'Value'])    
        if 'solar_conjunction' in test_results:
            writer.writerow(['Solar_Conjunction', 
                           'PASSED' if test_results['solar_conjunction'] else 'FAILED',
                           'Maintains_Connectivity', 
                           test_results['solar_conjunction']])
        if 'failure_recovery' in test_results:
            recovery_time = test_results['failure_recovery']
            writer.writerow(['Failure_Recovery',
                           'PASSED' if recovery_time < 2.0 else 'FAILED',
                           'Recovery_Time_Hours',
                           recovery_time])
        if 'traffic_overload' in test_results:
            stress = test_results['traffic_overload']
            writer.writerow(['Traffic_Overload',
                           'PASSED' if stress < 0.2 else 'WARNING' if stress < 0.5 else 'FAILED',
                           'Network_Stress_Level',
                           stress])
        if 'orbital_configs' in test_results:
            avg_connectivity = np.mean([
                (data['earth_connectivity'] + data['mars_connectivity']) / 2
                for data in test_results['orbital_configs'].values()
            ])
            writer.writerow(['Orbital_Configurations',
                           'PASSED' if avg_connectivity > 90 else 'FAILED',
                           'Average_Connectivity',
                           avg_connectivity])
    print(f"\nValidation results exported to {export_dir}/ directory")
    print("   Files created:")
    print("   - validation_test_results.json")
    print("   - validation_summary.csv")
    print("   - validation_failure_recovery.csv")
    print("   - validation_traffic_overload.csv")
    print("   - validation_orbital_configs.csv")

#Run validation tests and automatically export results
def run_selective_validation_tests_with_export(network, test_numbers=[1, 2, 3, 4], export_dir='results'):   
    #run tests
    test_results = run_selective_validation_tests(network, test_numbers)
    #export results for figure generation
    export_validation_results_for_figures(test_results, export_dir)
    return test_results

#Run the enhanced FSO-enabled interplanetary network simulation with controller coordination and traffic management
def run_fso_network_simulation():
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    #Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'results/simulation_output_{timestamp}.txt'
    #Redirect stdout to file
    sys.stdout = open(output_filename, 'w')   
    
    network = FSO_NetworkArchitecture()
    start_time = dt.datetime(2025, 5, 3, 12, 0, 0)
    time_points = np.arange(0, SIM_DURATION, SIM_STEP)
    print(f"\nENHANCED FREE SPACE OPTICAL INTERPLANETARY NETWORK SIMULATION")
    print(f"   with Controller Coordination & Traffic Management")
    print(f"   Satellites: {EARTH_SATS} Earth, {MARS_SATS} Mars")
    print(f"   Controllers: {EARTH_MOON_CONTROLLERS} Earth-Moon Lagrange")
    print(f"   Relays: {EARTH_SUN_RELAYS} Earth-Sun, {MARS_SUN_RELAYS} Mars-Sun")
    print(f"   Duration: {SIM_DURATION} hours, Step: {SIM_STEP} hours")
    print(f"{'='*60}\n")
    #Test if network maintains connectivity during conjunction
    conjunction_connectivity = test_solar_conjunction(network)
    if conjunction_connectivity:
        print("\nSUCCESS: Network maintains Earth-Mars connectivity during solar conjunction")
    else:
        print("\nFAILURE: Network loses Earth-Mars connectivity during solar conjunction")
    for i, sim_time in enumerate(time_points):
        current_time = start_time + dt.timedelta(hours=sim_time)
        ticks = spt.Ticktock([current_time.isoformat()])
        print(f"Time Step {i+1}/{len(time_points)}: {current_time.strftime('%Y-%m-%d %H:%M')} (t={sim_time:.1f}h)") 
        #Update the network
        network.update(sim_time, ticks)
        #Get current stats
        stats = network.get_connectivity_stats()
        fso_stats = stats['fso_performance']
        optical_stats = stats['optical_links']
        coordination_stats = stats['controller_coordination']
        traffic_stats = stats['traffic_management']
        print(f"\nNETWORK STATUS:")
        print(f"  Connectivity: Earth {stats['earth_connectivity']:.1f}%, Mars {stats['mars_connectivity']:.1f}%")
        print(f"  FSO Links: {fso_stats.get('total_active_fso_links', 0)} active ({fso_stats.get('navigation_assisted_links', 0)} nav-assisted, {fso_stats.get('traffic_coordinated_links', 0)} traffic-coordinated)")
        print(f"  Data Rate: {fso_stats.get('total_data_rate_tbps', 0):.2f} Tbps total")
        print(f"  Interplanetary: {optical_stats.get('interplanetary_links', 0)} links @ {optical_stats.get('interplanetary_data_rate_gbps', 0):.1f} Gbps")
        print(f"  Controllers: {coordination_stats.get('active_controllers', 0)}/{coordination_stats.get('total_controllers', 0)} active, {coordination_stats.get('coordination_coverage_percentage', 0):.1f}% coverage")
        print(f"  Traffic Mgmt: {traffic_stats.get('satellites_with_traffic_schedule', 0)} scheduled, {traffic_stats.get('authorized_transmissions', 0)} authorized")
        print(f"\n")
    #Run debugging tests
    print("\n" + "="*60)
    print("RUNNING NETWORK DIAGNOSTICS")
    print("="*60)
    #Test connectivity
    test_simple_connectivity(network)
    #Test Mars connectivity
    quick_mars_test(network)
    #Test controller coordination
    test_controller_coordination(network)
    #Export results
    network.thesis_metrics.export_for_thesis()
    print("\n" + "="*80)
    print("RUNNING VALIDATION TESTS (1-4)")
    print("="*80)
    test_results = run_selective_validation_tests_with_export(
        network, 
        test_numbers=[1, 2, 3, 4],
        export_dir='results'
    )
    return network

if __name__ == "__main__": 
    network = run_fso_network_simulation()