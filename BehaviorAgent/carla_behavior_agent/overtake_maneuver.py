"""
Questo modulo contiene la classe OvertakeManeuver che gestisce la logica
di sorpasso per l'agente di comportamento.
"""

import carla
import math
from misc import *

class OvertakingManeuver:
    """
    Classe che gestisce la logica di sorpasso per l'agente di comportamento.
    Si occupa di calcolare il percorso di sorpasso, verificare le condizioni di sicurezza
    e tenere traccia dello stato del sorpasso.
    """

    def __init__(self, vehicle, world, map_inst, local_planner, sampling_resolution):
        """
        Inizializza la classe OvertakeManeuver.
        
        :param vehicle: il veicolo controllato dall'agente
        :param world: il mondo CARLA
        :param map_inst: l'istanza della mappa
        :param local_planner: il pianificatore locale dell'agente
        :param sampling_resolution: la risoluzione di campionamento per il percorso
        """
        self._vehicle = vehicle
        self._world = world
        self._map = map_inst
        self._local_planner = local_planner
        self._sampling_resolution = sampling_resolution
        
        # Attributi per l'overtake
        self._overtake_cnt = 0
        self._in_overtake = False
        self._overtake_ego_distance = 0
    
    @property
    def in_overtake(self) -> bool:
        """Indica se l'agente sta eseguendo una manovra di sorpasso."""
        return self._in_overtake
    
    @property
    def overtake_cnt(self) -> int:
        """Restituisce il contatore del sorpasso."""
        return self._overtake_cnt
    
    @overtake_cnt.setter
    def overtake_cnt(self, value: int) -> None:
        """Imposta il contatore del sorpasso."""
        self._overtake_cnt = value
        
    @in_overtake.setter
    def in_overtake(self, value: bool) -> None:
        """Imposta lo stato del sorpasso."""
        self._in_overtake = value

    @property
    def overtake_ego_distance(self) -> float:
        """Restituisce la distanza che l'agente deve percorrere per completare il sorpasso."""
        return self._overtake_ego_distance
        
    def _get_distance_other_lane(self, actor: carla.Actor, max_distance: float = 30) -> float:
        """
        Calcola la distanza che l'agente deve percorrere per sorpassare gli ostacoli.
        
        :param actor: attore che l'agente deve sorpassare
        :param max_distance: distanza massima di ricerca
        :return: distanza da percorrere sulla corsia adiacente
        """
        # Ottieni la lunghezza del veicolo da sorpassare
        actor_length = actor.bounding_box.extent.x * 2
        
        # Distanza di sicurezza minima dopo il sorpasso
        safety_buffer = 5.0  # metri
        
        # Inizializza la distanza di sorpasso con solo l'attore principale + buffer
        distance_other_lane = actor_length + safety_buffer

        # Ottieni la lista dei veicoli nella simulazione
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        actor_wp = self._map.get_waypoint(actor.get_location())
        
        vehicle_list = [
            v for v in vehicle_list 
            if v.id != actor.id and v.id != self._vehicle.id and dist(v, actor) < max_distance \
            and self._map.get_waypoint(v.get_location()).lane_id == actor_wp.lane_id
        ]

        # Filtra solo i veicoli effettivamente fermi e davanti all'attore principale
        relevant_vehicles = []
        for v in vehicle_list:
            _, is_stopped = self._is_vehicle_legitimately_stopped(v)
            if is_stopped:
                # Controlla se il veicolo è davanti all'attore nella direzione di marcia
                if self._is_vehicle_ahead(actor, v):
                    relevant_vehicles.append(v)
        
        if not relevant_vehicles:
            return distance_other_lane
        
        # Ordina i veicoli per distanza dall'attore principale
        relevant_vehicles.sort(key=lambda v: dist(actor, v))
        
        # Considera solo i veicoli in una catena continua (senza gap troppo grandi)
        max_gap = 12.0  # metri - gap massimo per considerare i veicoli in sequenza
        previous_vehicle = actor
        continuous_chain_distance = 0
        
        for v in relevant_vehicles:
            
            if is_within_distance(target_transform=v.get_transform(), reference_transform=previous_vehicle.get_transform(), max_distance=max_distance, angle_interval=[0, 60]):
                vehicle_distance = compute_distance_from_center(actor1=previous_vehicle, actor2=v, distance=dist(v, previous_vehicle))
            else:
                continue
            # Se il gap è troppo grande, interrompi la catena
            # if vehicle_distance > max_gap:
            #     break
                
            # Aggiungi la distanza del veicolo alla catena
            continuous_chain_distance += vehicle_distance + (v.bounding_box.extent.x * 2)
            previous_vehicle = v
        
        # La distanza finale è la lunghezza dell'attore principale + catena continua (removed buffer for now)
        final_distance = actor_length + continuous_chain_distance + safety_buffer
        
        # Limita la distanza massima per evitare sorpassi eccessivamente lunghi
        max_overtake_distance = 35.0  # metri massimi sulla corsia opposta
        
        return min(final_distance, max_overtake_distance)

    def _is_vehicle_ahead(self, reference_vehicle: carla.Actor, target_vehicle: carla.Actor) -> bool:
        """
        Verifica se target_vehicle è davanti a reference_vehicle nella direzione di marcia.
        
        :param reference_vehicle: veicolo di riferimento
        :param target_vehicle: veicolo da controllare
        :return: True se target_vehicle è davanti, False altrimenti
        """
        ref_transform = reference_vehicle.get_transform()
        target_location = target_vehicle.get_location()
        
        # Vettore dal veicolo di riferimento al target
        to_target = target_location - ref_transform.location
        
        # Vettore in avanti del veicolo di riferimento
        forward_vector = ref_transform.get_forward_vector()
        
        # Prodotto scalare per determinare se è davanti
        dot_product = to_target.x * forward_vector.x + to_target.y * forward_vector.y
        
        return dot_product > 0


    def run_overtake_step(
        self, 
        object_to_overtake, 
        ego_vehicle_wp,
        distance_same_lane = 2.0,  # Aumentato leggermente per sicurezza
        distance_other_lane = 0.0,
        distance_from_object = 15.0,  # Ridotto da 18 a 15
        speed_limit = 50
    ):
        """
        Esegue la logica per il sorpasso.
        
        :param object_to_overtake: l'oggetto da sorpassare (carla.Actor)
        :param ego_vehicle_wp: il waypoint del veicolo (carla.Waypoint)
        :param distance_same_lane: la distanza da percorrere sulla stessa corsia prima di cambiare (float)
        :param distance_other_lane: la distanza da percorrere sull'altra corsia (float)
        :param distance_from_object: la distanza dall'oggetto da sorpassare (float)
        :param speed_limit: il limite di velocità della strada (float)
        :return: il percorso di sorpasso o None se non è possibile
        """
        # Ottieni la distanza sull'altra corsia se non è fornita
        if not distance_other_lane:
            distance_other_lane = self._get_distance_other_lane(object_to_overtake, 30)  # Ridotto da 30 a 25

        # Ottieni la lunghezza totale del veicolo
        vehicle_length = self._vehicle.bounding_box.extent.x 
        lane_width = ego_vehicle_wp.lane_width

        # Calcola la distanza di sorpasso che l'agente deve percorrere
        self._overtake_ego_distance, hypotenuse = self.get_overtake_distance(
            vehicle_length, lane_width, distance_same_lane, distance_other_lane, distance_from_object
        )

        # Calcola il tempo necessario per il sorpasso
        overtake_time = self.get_overtake_time(ego_vehicle=self._vehicle, overtake_distance=self._overtake_ego_distance)
        
        # Calcola la distanza percorsa da un veicolo nella corsia opposta durante il sorpasso
        opposite_vehicle_distance = (overtake_time * speed_limit / 3.6)
        search_distance = max(self._overtake_ego_distance + opposite_vehicle_distance,30.0)

        # Controlla se c'è un veicolo nella corsia opposta che può ostacolare il sorpasso
        opposing_vehicle = self._opposite_vehicle(ego_wp=ego_vehicle_wp, search_distance=search_distance)

        # Verifica più rigorosa delle condizioni di sicurezza
        collision = False
        wp_in_junction = False
        
        try:
            # Ottieni il prossimo waypoint del veicolo a distanza 'overtake_ego_distance'
            next_ego_wps = ego_vehicle_wp.next(self._overtake_ego_distance)
            if next_ego_wps:
                next_ego_wp = next_ego_wps[0]
                wp_in_junction = next_ego_wp.is_junction
                
                if opposing_vehicle:
                    next_opposing_wps = self._map.get_waypoint(opposing_vehicle.get_location()).next(opposite_vehicle_distance)
                    if next_opposing_wps:
                        next_opposing_wp = next_opposing_wps[0]
                        
                        collision = not is_within_distance(
                            target_transform=next_opposing_wp.transform, 
                            reference_transform=next_ego_wp.transform, 
                            max_distance=25,  # Ridotto da 30 a 25 per essere più cautelativo
                            angle_interval=[0, 90]
                        )
        except Exception as e:
            # In caso di errore, sii cautelativo
            collision = True if opposing_vehicle else False
            
        # Condizioni più stringenti per iniziare il sorpasso
        safe_to_overtake = (
            not self._overtake_cnt and 
            not collision and 
            not wp_in_junction and
            self._overtake_ego_distance < 70.0  # Limite massimo di distanza per il sorpasso
        )
                        
        if safe_to_overtake:
            # Genera il percorso per sorpassare il veicolo
            overtake_path = self._local_planner._generate_lane_change_path(
                waypoint=ego_vehicle_wp,
                direction='left',
                distance_same_lane=distance_same_lane,
                distance_other_lane=distance_other_lane,
                lane_change_distance=hypotenuse,
                check=False, 
                step_distance=self._sampling_resolution,
            )
            
            # Se il percorso non è generato, ritorna None
            if not overtake_path:
                return None
                    
            # Aggiorna il contatore del sorpasso con un tempo più realistico
            self._overtake_cnt = max(
                int(round(overtake_time) / self._world.get_snapshot().timestamp.delta_seconds),
                30  # Minimo 30 frame per evitare sorpassi troppo veloci
            )
            
            # Aggiorna lo stato del sorpasso
            self._in_overtake = True
            
            return overtake_path
        
        return None

    def _opposite_vehicle(self, ego_wp: carla.Waypoint = None, search_distance: float = 30) -> carla.Actor:
        """
        Restituisce il veicolo nella corsia opposta.
        
        :param ego_wp: waypoint del veicolo
        :param search_distance: distanza massima di ricerca
        :return: attore del veicolo nella corsia opposta
        """
        def extend_bounding_box(actor: carla.Actor) -> carla.Transform:
            """
            Calcola il punto estremo dell'attore nella posizione specificata.
            
            :param actor: attore
            :return: transform del punto estremo dell'attore
            """
            # Ottieni il waypoint dell'attore
            wp = self._map.get_waypoint(actor.get_location())
            transform = wp.transform
            # Ottieni il vettore in avanti dell'attore
            forward_vector = transform.get_forward_vector()
            # Estendi il bounding box dell'attore
            extent = actor.bounding_box.extent.x
            location = carla.Location(x=extent * forward_vector.x, y=extent * forward_vector.y)
            transform.location += location
            return transform
        
        # Ottieni tutti i veicoli vicini all'ego
        vehicle_list = self._get_ordered_vehicles(self._vehicle, search_distance)
        # Filtra i veicoli nella stessa corsia dell'ego
        vehicle_list = [
            v for v in vehicle_list 
            if self._map.get_waypoint(v.get_location()).lane_id == ego_wp.lane_id * -1
        ]

        if not vehicle_list:
            return None

        # Estendi il bounding box del veicolo ego
        ego_front_transform = extend_bounding_box(self._vehicle)

        for vehicle in vehicle_list:
            # Estendi il bounding box del veicolo
            target_front_transform = extend_bounding_box(vehicle)
            # Controlla se il veicolo è nella corsia opposta
            if is_within_distance(
                target_front_transform, 
                ego_front_transform, 
                search_distance, 
                angle_interval=[0, 90]
            ):
                return vehicle
    
        return None

    @staticmethod
    def get_overtake_distance(
        vehicle_length, 
        lane_width, 
        distance_same_lane, 
        distance_other_lane,
        distance_from_obstacle
    ):
        """
        Calcola la distanza che il veicolo deve percorrere per sorpassare.
        
        :param vehicle_length: lunghezza del veicolo
        :param lane_width: larghezza della corsia
        :param distance_same_lane: distanza sulla stessa corsia
        :param distance_other_lane: distanza sull'altra corsia
        :param distance_from_obstacle: distanza dall'ostacolo
        :return: distanza di sorpasso e ipotenusa
        """
        # Calcola l'ipotenusa per il cambio di corsia (più realistica)
        # Usa un angolo di cambio corsia più graduale
        lane_change_angle_distance = 8.0  # metri per cambiare corsia gradualmente
        hypotenuse = math.sqrt(vehicle_length**2 + lane_width**2)
        
        # Calcola la distanza totale con parametri più conservativi
        overtake_distance = (
            distance_from_obstacle +  # Riduci la distanza iniziale
            distance_same_lane + 
            hypotenuse + 
            distance_other_lane + 
            hypotenuse
        )
        
        return overtake_distance, hypotenuse

    @staticmethod
    def get_overtake_time(ego_vehicle: carla.Vehicle, overtake_distance: float) -> float:
        """
        Calcola il tempo necessario per il sorpasso.
        
        :param ego_vehicle: veicolo
        :param overtake_distance: distanza di sorpasso
        :return: tempo di sorpasso in secondi
        """
        # Ottieni la velocità attuale del veicolo
        v0 = get_speed(ego_vehicle) / 3.6  # m/s
    
        # Accelerazione del veicolo
        a = 3.1 # m/s^2
        
        # Calcola il tempo di sorpasso
        overtake_time = (-v0 + math.sqrt(v0 ** 2 + 2 * a * overtake_distance)) / a
        
        return overtake_time
    
    def update_overtake_plan(self, overtake_path: list, speed_multiplier: float = 2.0) -> None:
        """
        Aggiorna il piano di sorpasso dell'agente.
        
        :param overtake_path: percorso di sorpasso
        :param speed_multiplier: moltiplicatore di velocità per il sorpasso
        """
        # Imposta il piano di sorpasso - questa chiamata ora aggiornerà anche la coda di waypoint
        self._local_planner.set_overtake_plan(
            overtake_plan=overtake_path, 
            overtake_distance=self._overtake_ego_distance
        )
        
        # Aggiorna la velocità dell'agente
        current_speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(current_speed_limit * speed_multiplier)
        
    def update_counts(self):
        """
        Aggiorna i contatori del sorpasso.
        Dovrebbe essere chiamato ad ogni step.
        """
        if self._overtake_cnt > 0:
            self._overtake_cnt -= 1
            return True
        else:
            self._in_overtake = False
            return False
            
    def _get_ordered_vehicles(self, vehicle, max_distance):
        """
        Ottiene i veicoli ordinati per distanza dal veicolo dato.
        
        :param vehicle: veicolo di riferimento
        :param max_distance: distanza massima
        :return: lista di veicoli ordinati
        """
        # Implementazione del metodo che originariamente era nella classe BehaviorAgent
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        vehicle_list = [v for v in vehicle_list if v.id != vehicle.id]
        
        # Calcola la distanza di ogni veicolo dal veicolo di riferimento
        vehicles_with_distances = [(v, dist(vehicle, v)) for v in vehicle_list if dist(vehicle, v) < max_distance]
        
        # Ordina i veicoli per distanza
        vehicles_with_distances.sort(key=lambda x: x[1])
        
        return [v for v, d in vehicles_with_distances]
    
    def _is_vehicle_legitimately_stopped(self, vehicle):
        """
        Verifica se un veicolo è legittimamente fermo.
        
        :param vehicle: veicolo da verificare
        :return: waypoint e booleano che indica se il veicolo è fermo
        """
        # Questa implementazione dipende dall'implementazione originale nella classe BehaviorAgent
        # Dovrai adattarla in base a come è implementata originariamente
        # Questo è solo un esempio
        vehicle_wp = self._map.get_waypoint(vehicle.get_location())
        
        # Verifica se il veicolo è fermo
        is_stopped = get_speed(vehicle) < 1.0  # Supponiamo che sia fermo se la velocità è inferiore a 1 km/h
        
        return vehicle_wp, is_stopped