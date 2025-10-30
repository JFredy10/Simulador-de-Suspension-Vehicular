import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re

# Se requiere instalar: pip install ttkthemes
try:
    from ttkthemes import ThemedTk
except ImportError:
    messagebox.showerror("Error de Dependencia", 
                         "No se encontró la biblioteca 'ttkthemes'.\n\n"
                         "Por favor, instálala abriendo una terminal y ejecutando:\n"
                         "pip install ttkthemes")
    exit()


# === Funciones de Simulación ===

def quarter_car_model(Y, t, ms, mu, Ks, Kt, C, A_step, T_step):
    """
    Sistema de EDOs de primer orden para el modelo de un cuarto de vehículo.
    Y = [zs, z_dot_s, zu, z_dot_u]
    """
    zs, z_dot_s, zu, z_dot_u = Y
    
    # Excitación: Desplazamiento de la carretera función de bache
    z_r_t = A_step * (t >= T_step)
    
    # EDO para la masa suspendida (ms)
    zs_ddot = (1/ms) * (-Ks * (zs - zu) - C * (z_dot_s - z_dot_u))
    
    # EDO para la masa no suspendida (mu)
    zu_ddot = (1/mu) * (Ks * (zs - zu) + C * (z_dot_s - z_dot_u) - Kt * (zu - z_r_t))
    
    return [z_dot_s, zs_ddot, z_dot_u, zu_ddot]

def simular_y_obtener_aceleracion(ms, mu, Ks, Kt, C, Tiempo, A_step, T_step):
    """Resuelve las EDOs y calcula la aceleración de la masa suspendida (zs_ddot)."""
    Y0 = [0.0, 0.0, 0.0, 0.0]
    
    try:
        sol = odeint(quarter_car_model, Y0, Tiempo, args=(ms, mu, Ks, Kt, C, A_step, T_step))
    except Exception as e:
        messagebox.showerror("Error de Integración", f"No se pudo resolver el sistema de EDOs: {e}")
        return Tiempo, np.zeros_like(Tiempo) 
    
    zs, z_dot_s, zu, z_dot_u = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]
    
    # Cálculo de la aceleración (z''_s)
    Acc_ms = (1/ms) * (-Ks * (zs - zu) - C * (z_dot_s - z_dot_u))
    return Tiempo, Acc_ms

# === CLASE DE LA INTERFAZ GRÁFICA ===

class SuspensionSimulatorGUI:
    def __init__(self, master):
        self.master = master
        master.title(" Simulador de Suspensión Vehicular ")
        master.geometry("1450x900") 
        
        style = ttk.Style()
        style.configure("TLabel", font=('Arial', 10))
        style.configure("TButton", font=('Arial', 11, 'bold'))
        style.configure("Title.TLabel", font=('Arial', 10, 'bold', 'italic'), foreground='#003366') 
        style.configure("Subtitle.TLabel", font=('Arial', 9, 'italic'), foreground='#6c757d') 

        # Variables de entrada y valores por defecto
        self.vars = {
            'ms_fijo': tk.StringVar(value="250.0"), 
            'mu_fijo': tk.StringVar(value="40.0"), 
            'kt_fijo': tk.StringVar(value="170000.0"),
            'a_step': tk.StringVar(value="0.1"),
            't_step': tk.StringVar(value="0.5"),
            't_sim': tk.StringVar(value="4.0"),
            'velocidad': tk.StringVar(value="30.0"), 
            'ms_fijo_C': tk.StringVar(value="250.0"),
            'Ks_fijo_C': tk.StringVar(value="16195.0"),
            'C_valores': tk.StringVar(value="800, 1300, 1800, 2500, 3000"),
            'C_fijo_Ks': tk.StringVar(value="1300.0"),
            'ms_fijo_Ks': tk.StringVar(value="250.0"),
            'Ks_valores': tk.StringVar(value="10000, 12291, 14583, 16874, 19165"),
            'C_fijo_ms': tk.StringVar(value="1300.0"),
            'Ks_fijo_ms': tk.StringVar(value="16195.0"),
            'ms_valores': tk.StringVar(value="100.0, 137.5, 175.0, 212.5, 250.0"),
        }

        self.create_widgets()
    
    def create_widgets(self):
        # 1. Marco principal
        main_frame = ttk.Frame(self.master, padding="15")
        main_frame.pack(fill='both', expand=True)

        # 2. Marco para la entrada de parámetros (Izquierda)
        input_frame = ttk.Frame(main_frame, padding="10", relief=tk.FLAT)
        input_frame.grid(row=0, column=0, sticky="ns", padx=15, pady=10)
        
        # 3. Marco para las gráficas (Derecha)
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=0, column=1, sticky="nsew", padx=15, pady=10)
        
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # --- Sub-marcos para entradas (Panel Izquierdo) ---
        self.create_fixed_params_entries(input_frame)
        
        # === CAMBIO REALIZADO AQUÍ ===
        # Se movió el botón a esta posición para asegurar visibilidad
        button_container = ttk.Frame(input_frame)
        button_container.pack(fill='x', pady=(20, 15)) # Aumenté el padding inferior

        ttk.Button(button_container, text="EJECUTAR SIMULACIÓN", command=self.run_simulation, 
                   style='TButton', cursor="hand2").pack(fill='x', ipady=5)
        # === FIN DEL CAMBIO ===

        self.create_variation_params_entries(input_frame)
        
        # --- Área de Gráficas (Inicialización) ---
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 9)) 
        self.fig.suptitle("Resultados de Aceleración Vertical de la Masa Suspendida", fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
    def create_fixed_params_entries(self, parent):
        fixed_params_frame = ttk.LabelFrame(parent, text="⚙️ Parámetros Fijos del Modelo y Excitación", padding="10")
        fixed_params_frame.pack(fill='x', pady=5)
        
        labels_fixed = {
            'ms_fijo': "Masa Susp. (Ms) [kg]:",
            'mu_fijo': "Masa No Susp. (mu) [kg]:",
            'kt_fijo': "Rigidez Neumático (Kt) [N/m]:",
        }
        labels_excitation = {
            'a_step': "Amplitud Bache (A_step) [m]:",
            't_step': "Tiempo de Bache (T_step) [s]:",
            't_sim': "Tiempo Simulación (T_sim) [s]:",
            'velocidad': "Velocidad [km/h]:",
        }
        
        ttk.Label(fixed_params_frame, text="-- Componentes del Vehículo --", style='Subtitle.TLabel').grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
        r = 1
        for key, label_text in labels_fixed.items():
            ttk.Label(fixed_params_frame, text=label_text).grid(row=r, column=0, sticky="w", pady=3, padx=5)
            ttk.Entry(fixed_params_frame, textvariable=self.vars[key], width=18).grid(row=r, column=1, sticky="ew", padx=5, pady=3)
            r += 1
            
        ttk.Label(fixed_params_frame, text="-- Parámetros de Excitación (Bache) --", style='Subtitle.TLabel').grid(row=r, column=0, columnspan=2, sticky="w", pady=(10, 5))
        r += 1
        for key, label_text in labels_excitation.items():
            ttk.Label(fixed_params_frame, text=label_text).grid(row=r, column=0, sticky="w", pady=3, padx=5)
            ttk.Entry(fixed_params_frame, textvariable=self.vars[key], width=18).grid(row=r, column=1, sticky="ew", padx=5, pady=3)
            r += 1

    def create_variation_params_entries(self, parent):
        variation_params_frame = ttk.LabelFrame(parent, text="Rangos de Variación de Parámetros", padding="10")
        variation_params_frame.pack(fill='x', pady=10)
        
        ttk.Label(variation_params_frame, text="Ingrese valores a variar separados por coma", 
                  font=('Arial', 9, 'italic')).grid(row=0, column=0, columnspan=2, sticky="w", pady=5, padx=5)

        # Sección C
        ttk.Label(variation_params_frame, text="-- 1. Variación de Coeficiente de Amortiguación (C) --", style='Title.TLabel').grid(row=1, column=0, columnspan=2, pady=(10, 5), sticky="w", padx=5)
        self.add_entry(variation_params_frame, "C_valores", "Valores C [Ns/m]:", 2)
        self.add_entry(variation_params_frame, "ms_fijo_C", "Ms fijo [kg]:", 3)
        self.add_entry(variation_params_frame, "Ks_fijo_C", "Ks fijo [N/m]:", 4)

        # Sección Ks
        ttk.Label(variation_params_frame, text="-- 2. Variación de Rigidez del Muelle (Ks) --", style='Title.TLabel').grid(row=5, column=0, columnspan=2, pady=(10, 5), sticky="w", padx=5)
        self.add_entry(variation_params_frame, "Ks_valores", "Valores Ks [N/m]:", 6)
        self.add_entry(variation_params_frame, "C_fijo_Ks", "C fijo [Ns/m]:", 7)
        self.add_entry(variation_params_frame, "ms_fijo_Ks", "Ms fijo [kg]:", 8)
        
        # Sección Ms
        ttk.Label(variation_params_frame, text="-- 3. Variación de Masa Suspendida (Ms) --", style='Title.TLabel').grid(row=9, column=0, columnspan=2, pady=(10, 5), sticky="w", padx=5)
        self.add_entry(variation_params_frame, "ms_valores", "Valores Ms [kg]:", 10)
        self.add_entry(variation_params_frame, "C_fijo_ms", "C fijo [Ns/m]:", 11)
        self.add_entry(variation_params_frame, "Ks_fijo_ms", "Ks fijo [N/m]:", 12)

    def add_entry(self, parent, key, label_text, row):
        """Función auxiliar para añadir etiquetas y campos de entrada."""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", pady=2, padx=5)
        ttk.Entry(parent, textvariable=self.vars[key], width=30).grid(row=row, column=1, sticky="ew", padx=5, pady=2)

    def validate_and_get_params(self):
        """Valida y convierte todos los parámetros de string a float/list."""
        params = {}
        try:
            # 1. Parámetros de un solo valor (Flotantes)
            single_value_keys = [k for k in self.vars.keys() if 'valores' not in k]
            for key in single_value_keys:
                value = float(self.vars[key].get().replace(',', '.')) 
                if value <= 0 and key not in ['a_step', 't_step']: 
                    messagebox.showerror("Error de Validación", f"El parámetro '{key}' debe ser positivo. Valor ingresado: {value}")
                    return None
                params[key] = value

            # 2. Parámetros de lista (Listas de flotantes positivos)
            list_keys = [k for k in self.vars.keys() if 'valores' in k]
            for key in list_keys:
                values_str = self.vars[key].get()
                str_list = [s.strip().replace(',', '.') for s in re.split(r'[;,]', values_str) if s.strip()]
                
                if not str_list:
                    messagebox.showerror("Error de Validación", f"La lista de valores para '{key}' no puede estar vacía.")
                    return None

                float_list = []
                for v_str in str_list:
                    value = float(v_str)
                    if value <= 0:
                        messagebox.showerror("Error de Validación", f"Todos los valores en la lista '{key}' deben ser positivos. Valor problemático: {value}")
                        return None
                    float_list.append(value)
                params[key] = float_list
                
            return params
            
        except ValueError as e:
            messagebox.showerror("Error de Formato", f"Asegúrese de que todos los campos contengan valores numéricos válidos (use '.' o ',' para decimales). Error: {e}")
            return None
        except Exception as e:
            messagebox.showerror("Error Desconocido", f"Ocurrió un error inesperado durante la validación: {e}")
            return None

    def run_simulation(self):
        params = self.validate_and_get_params()
        if not params: return 

        Tiempo = np.linspace(0, params['t_sim'], 1000)
        self.clear_plots()
        
        global_results = {'C': {'max': -np.inf, 'min': np.inf}, 
                          'Ks': {'max': -np.inf, 'min': np.inf}, 
                          'ms': {'max': -np.inf, 'min': np.inf}}
        
        try:
            self.plot_C(params, Tiempo, global_results)
            self.plot_Ks(params, Tiempo, global_results)
            self.plot_ms(params, Tiempo, global_results)
            self.plot_optimum(params, Tiempo)
            
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
            self.canvas.draw()
            
            self.show_results_summary(params, global_results)
        
        except Exception as e:
            messagebox.showerror("Error de Simulación", f"Ocurrió un error durante la ejecución de la simulación: {e}")

    # --- Métodos de Ploteo ---
    
    def clear_plots(self):
        """Limpia  las etiquetas ."""
        for ax in self.axs.flat:
            ax.clear()
            ax.set_xlabel('Tiempo [s]')
            ax.set_ylabel('Aceleración [m/s²]')
            ax.grid(True)

    def plot_C(self, params, Tiempo, global_results):
        """Gráfica 1: Variación de C (Amortiguamiento)"""
        ax = self.axs[0, 0]
        ax.set_title('1. Efecto del Coeficiente de Amortiguamiento (C)')
        
        N = len(params['C_valores'])
        color_map = plt.colormaps['viridis'].resampled(N)
        
        for i, C_actual in enumerate(params['C_valores']):
            ms = params['ms_fijo_C']
            Ks = params['Ks_fijo_C']
            
            C_critico = 2 * np.sqrt(ms * Ks)
            zeta = C_actual / C_critico 
            
            if zeta < 0.98: tipo = f'Subamortiguado (ζ={zeta:.3f})'
            elif zeta > 1.02: tipo = f'Sobreamortiguado (ζ={zeta:.3f})'
            else: tipo = f'Crítico (ζ≈1)'

            T, Acc = simular_y_obtener_aceleracion(ms, params['mu_fijo'], Ks, params['kt_fijo'], C_actual, Tiempo, params['a_step'], params['t_step'])
            
            global_results['C']['max'] = max(global_results['C']['max'], np.max(Acc))
            global_results['C']['min'] = min(global_results['C']['min'], np.min(Acc))
            
            ax.plot(T, Acc, linewidth=1.5, label=f'C={C_actual:.0f} Ns/m ({tipo})', color=color_map(i/N))

        ax.legend(fontsize=8, loc='upper right') 
        
    def plot_Ks(self, params, Tiempo, global_results):
        """Gráfica 2: Variación de Ks (Rigidez Muelle)"""
        ax = self.axs[0, 1]
        ax.set_title('2. Efecto de la Rigidez del Muelle (Ks)')
        
        N = len(params['Ks_valores'])
        color_map = plt.colormaps['plasma'].resampled(N)
        
        for i, Ks_actual in enumerate(params['Ks_valores']):
            ms = params['ms_fijo_Ks']
            C = params['C_fijo_Ks']
            
            T, Acc = simular_y_obtener_aceleracion(ms, params['mu_fijo'], Ks_actual, params['kt_fijo'], C, Tiempo, params['a_step'], params['t_step'])
            
            global_results['Ks']['max'] = max(global_results['Ks']['max'], np.max(Acc))
            global_results['Ks']['min'] = min(global_results['Ks']['min'], np.min(Acc))

            ax.plot(T, Acc, linewidth=1.5, color=color_map(i/N), label=f'Ks={Ks_actual:.0f} N/m')

        ax.legend(fontsize=8, loc='upper right')

    def plot_ms(self, params, Tiempo, global_results):
        """Gráfica 3: Variación de Ms (Masa Suspendida)"""
        ax = self.axs[1, 0]
        ax.set_title('3. Efecto de la Masa Suspendida (Ms)')
        
        N = len(params['ms_valores'])
        color_map = plt.colormaps['cividis'].resampled(N)
        
        for i, ms_actual in enumerate(params['ms_valores']):
            Ks = params['Ks_fijo_ms']
            C = params['C_fijo_ms']
            
            T, Acc = simular_y_obtener_aceleracion(ms_actual, params['mu_fijo'], Ks, params['kt_fijo'], C, Tiempo, params['a_step'], params['t_step'])
            
            global_results['ms']['max'] = max(global_results['ms']['max'], np.max(Acc))
            global_results['ms']['min'] = min(global_results['ms']['min'], np.min(Acc))

            ax.plot(T, Acc, linewidth=1.5, color=color_map(i/N), label=f'Ms={ms_actual:.1f} kg')

        ax.legend(fontsize=8, loc='upper right')

    def plot_optimum(self, params, Tiempo):
        """Gráfica 4: Simulación con Amortiguamiento Óptimo"""
        ax = self.axs[1, 1]
        
        m_s_optimo = params['ms_fijo_C']
        K_s_optimo = params['Ks_fijo_C']
        
        C_critico_optimo = 2 * np.sqrt(m_s_optimo * K_s_optimo)
        zeta_optimo = 0.707 
        C_optimo = zeta_optimo * C_critico_optimo
        
        T, Acc = simular_y_obtener_aceleracion(m_s_optimo, params['mu_fijo'], K_s_optimo, params['kt_fijo'], C_optimo, Tiempo, params['a_step'], params['t_step'])
        
        ax.plot(T, Acc, linewidth=2.0, color='b', label=f'C={C_optimo:.0f} Ns/m') 
        ax.set_title(f'4. Amortiguamiento Óptimo (ζ ≈ {zeta_optimo:.3f})')
        ax.legend(fontsize=8, loc='upper right') 

    def show_results_summary(self, params, global_results):
        """Muestra un resumen de los picos de aceleración en un messagebox."""
        m_s = params['ms_fijo_C']
        K_s = params['Ks_fijo_C']
        C_critico = 2 * np.sqrt(m_s * K_s)
        
        summary = f"--- Resumen del Sistema Base (Ms={m_s:.1f} kg, Ks={K_s:.0f} N/m) ---\n"
        summary += f"Amortiguamiento Crítico (C_critico, ζ=1): {C_critico:.1f} Ns/m\n"
        summary += f"Amortiguamiento Óptimo (ζ=0.707): {C_critico * 0.707:.1f} Ns/m\n\n"
        
        summary += "--- Picos de Aceleración (m/s²) ---\n"
        summary += f"1. Variación C: [Min: {global_results['C']['min']:.2f}, Max: {global_results['C']['max']:.2f}]\n"
        summary += f"2. Variación Ks: [Min: {global_results['Ks']['min']:.2f}, Max: {global_results['Ks']['max']:.2f}]\n"
        summary += f"3. Variación Ms: [Min: {global_results['ms']['min']:.2f}, Max: {global_results['ms']['max']:.2f}]\n"
        
        messagebox.showinfo("Resumen de Simulación", summary)

# === Punto de entrada principal ===
if __name__ == '__main__':
    try:
        root = ThemedTk(theme="yaru") 
        root.set_theme("yaru") 
        app = SuspensionSimulatorGUI(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error de Aplicación", 
                             f"No se pudo iniciar la interfaz gráfica. "
                             f"Asegúrese de que su entorno Python sea correcto.\n\n"
                             f"Error: {e}")
