import pickle
import os
import datetime
import cv2
from pathlib import Path

class StateManager:
    @staticmethod
    def save_application_state(gv, filepath):
        """
        Guarda el estado actual de la aplicación para poder restaurarlo en caso de cierre inesperado.
        
        Args:
            gv: Instancia de VGlobals con el estado de la aplicación
            filepath: Ruta donde se guardará el archivo de estado
        """
        try:
            # Crear diccionario con variables de estado esenciales
            state_dict = {
                # Información de videos
                'filenames': gv.filenames if hasattr(gv, 'filenames') else None,
                'filename': gv.filename if hasattr(gv, 'filename') else None,
                'frameactual': gv.frameactual if hasattr(gv, 'frameactual') else 0,
                'frameaux': gv.frameaux if hasattr(gv, 'frameaux') else 0,
                
                # Configuración
                'configuracion': gv.configuracion if hasattr(gv, 'configuracion') else None,
                'model_path': gv.model_path if hasattr(gv, 'model_path') else "src/models_data/10-3.pt",
                
                # Estado de selecciones
                'area_seleccionada': gv.area_seleccionada if hasattr(gv, 'area_seleccionada') else False,
                'entrada_salida_seleccionada': gv.entrada_salida_seleccionada if hasattr(gv, 'entrada_salida_seleccionada') else False,
                'conversion_seleccionada': gv.conversion_seleccionada if hasattr(gv, 'conversion_seleccionada') else False,
                
                # Coordenadas y parámetros de área
                'x1': gv.x1 if hasattr(gv, 'x1') else 0,
                'y1': gv.y1 if hasattr(gv, 'y1') else 0,
                'x2': gv.x2 if hasattr(gv, 'x2') else 640,
                'y2': gv.y2 if hasattr(gv, 'y2') else 480,
                'yinicio': gv.yinicio if hasattr(gv, 'yinicio') else 460,
                'yfinal': gv.yfinal if hasattr(gv, 'yfinal') else 20,
                'entrada_coord': gv.entrada_coord if hasattr(gv, 'entrada_coord') else None,
                'salida_coord': gv.salida_coord if hasattr(gv, 'salida_coord') else None,
                'direccion': gv.direccion if hasattr(gv, 'direccion') else 'arriba_a_abajo',
                'cte': gv.cte if hasattr(gv, 'cte') else 0,
                
                # Estado de seguimiento
                'hojas': gv.hojas if hasattr(gv, 'hojas') else [],
                'hojas_final': gv.hojas_final if hasattr(gv, 'hojas_final') else [],
                'hojas_final_sale': gv.hojas_final_sale if hasattr(gv, 'hojas_final_sale') else [],
                
                # Contadores
                'ID': gv.ID if hasattr(gv, 'ID') else -1,
                'valid_ID': gv.valid_ID if hasattr(gv, 'valid_ID') else 0,
                'total_hojas': gv.total_hojas if hasattr(gv, 'total_hojas') else 0,
                
                # Variables de tiempo para continuidad entre videos
                'garch': gv.garch if hasattr(gv, 'garch') else 0.0,
                'frameactual_total': gv.frameactual_total if hasattr(gv, 'frameactual_total') else 0,
                
                # Carpeta de guardado
                'carpeta_seleccionada': gv.carpeta_seleccionada if hasattr(gv, 'carpeta_seleccionada') else None,
                
                # Estado de pausa
                'paused': gv.paused if hasattr(gv, 'paused') else True,
                
                # Timestamp del guardado
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Guardar estado en archivo pickle
            with open(filepath, 'wb') as f:
                pickle.dump(state_dict, f)
            
            print(f"Estado guardado en {filepath}")
            return True
        
        except Exception as e:
            print(f"Error al guardar el estado: {str(e)}")
            return False
    
    @staticmethod
    def load_application_state(gv, filepath):
        """
        Carga el estado de la aplicación desde un archivo guardado.
        
        Args:
            gv: Instancia de VGlobals donde se cargará el estado
            filepath: Ruta del archivo de estado a cargar
            
        Returns:
            bool: True si se cargó correctamente, False en caso contrario
        """
        try:
            if not os.path.exists(filepath):
                print(f"No se encontró archivo de estado en {filepath}")
                return False
            
            # Cargar estado desde archivo pickle
            with open(filepath, 'rb') as f:
                state_dict = pickle.load(f)
            
            # Restaurar variables de estado
            # Información de videos
            if state_dict.get('filenames'):
                gv.filenames = state_dict['filenames']
            if state_dict.get('filename'):
                gv.filename = state_dict['filename']
            gv.frameactual = state_dict.get('frameactual', 0)
            gv.frameaux = state_dict.get('frameaux', 0)
            
            # Configuración
            if state_dict.get('configuracion'):
                gv.configuracion = state_dict['configuracion']
            if state_dict.get('model_path'):
                gv.model_path = state_dict['model_path']
            
            # Estado de selecciones
            gv.area_seleccionada = state_dict.get('area_seleccionada', False)
            gv.entrada_salida_seleccionada = state_dict.get('entrada_salida_seleccionada', False)
            gv.conversion_seleccionada = state_dict.get('conversion_seleccionada', False)
            
            # Coordenadas y parámetros de área
            gv.x1 = state_dict.get('x1', 0)
            gv.y1 = state_dict.get('y1', 0)
            gv.x2 = state_dict.get('x2', 640)
            gv.y2 = state_dict.get('y2', 480)
            gv.yinicio = state_dict.get('yinicio', 460)
            gv.yfinal = state_dict.get('yfinal', 20)
            gv.entrada_coord = state_dict.get('entrada_coord')
            gv.salida_coord = state_dict.get('salida_coord')
            gv.direccion = state_dict.get('direccion', 'arriba_a_abajo')
            gv.cte = state_dict.get('cte', 0)
            
            # Estado de seguimiento
            gv.hojas = state_dict.get('hojas', [])
            gv.hojas_final = state_dict.get('hojas_final', [])
            gv.hojas_final_sale = state_dict.get('hojas_final_sale', [])
            
            # Contadores
            gv.ID = state_dict.get('ID', -1)
            gv.valid_ID = state_dict.get('valid_ID', 0)
            gv.total_hojas = state_dict.get('total_hojas', 0)
            
            # Variables de tiempo para continuidad entre videos
            gv.garch = state_dict.get('garch', 0.0)
            gv.frameactual_total = state_dict.get('frameactual_total', 0)
            
            # Carpeta de guardado
            if state_dict.get('carpeta_seleccionada'):
                gv.carpeta_seleccionada = state_dict['carpeta_seleccionada']
            
            # Estado de pausa
            gv.paused = state_dict.get('paused', True)
            
            print(f"Estado cargado desde {filepath} (guardado el {state_dict.get('timestamp', 'desconocido')})")
            return True
            
        except Exception as e:
            print(f"Error al cargar el estado: {str(e)}")
            return False
    
    @staticmethod
    def get_default_state_path(gv):
        """Obtiene la ruta por defecto del archivo de estado"""
        if hasattr(gv, 'carpeta_seleccionada') and gv.carpeta_seleccionada:
            # Si hay una carpeta de guardado seleccionada, usar esa
            return os.path.join(gv.carpeta_seleccionada, "antnalyzer_state.pkl")
        else:
            # En caso contrario, usar el directorio de la aplicación
            return "antnalyzer_state.pkl"