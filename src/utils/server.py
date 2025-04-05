import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import time
from datetime import datetime
import os
import threading

def launch_dash(csv_file_path):
    app = dash.Dash(__name__)
    
    # Carica i dati iniziali se il file esiste
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        last_data_length = len(df)
    else:
        df = pd.DataFrame()
        last_data_length = 0
    
    def update_data():
        nonlocal df, last_data_length
        while True:
            try:
                if os.path.exists(csv_file_path):
                    new_df = pd.read_csv(csv_file_path)
                    if len(new_df) > len(df):
                        df = new_df
                        last_data_length = len(df)
            except Exception as e:
                print(f"Errore durante la lettura del file: {e}")
            time.sleep(0.3)
    
    update_thread = threading.Thread(target=update_data, daemon=True)
    update_thread.start()
    
    # Crea le figure iniziali
    def create_initial_figures():
        if df.empty:
            return go.Figure(), go.Figure()
        
        # Figura per il grafico Reward
        reward_fig = px.line(df, x='Training Steps', y='Reward',
                           title='Reward Progress During Training')
        reward_fig.update_layout(
            xaxis_title='Training Steps',
            yaxis_title='Reward',
            hovermode='x unified',
            height=600
        )
        
        # Figura per i componenti
        reward_components = [col for col in df.columns 
                           if col not in ['Training Steps'] 
                           and pd.api.types.is_numeric_dtype(df[col])]
        
        components_fig = go.Figure()
        for component in reward_components:
            components_fig.add_trace(go.Scatter(
                x=df['Training Steps'],
                y=df[component],
                mode='lines',
                name=component
            ))
        
        components_fig.update_layout(
            title='Reward Components',
            xaxis_title='Training Steps',
            yaxis_title='Valore',
            height=600,
            hovermode='x unified'
        )
        
        return reward_fig, components_fig
    
    initial_reward_fig, initial_components_fig = create_initial_figures()
    
    app.layout = html.Div(style={'height': '100vh', 'display': 'flex', 'flexDirection': 'column'}, children=[        
        dcc.Tabs(style={'flex': '0 0 auto'}, children=[
            dcc.Tab(label='Reward Chart', children=[
                dcc.Graph(
                    id='live-graph',
                    config={'scrollZoom': True},
                    figure=initial_reward_fig
                ),
                dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
            ]),
            
            dcc.Tab(label='Components Detail', children=[
                dcc.Graph(
                    id='components-graph',
                    figure=initial_components_fig
                ),
                dcc.Interval(id='interval-components', interval=1000, n_intervals=0)
            ]),
            
            dcc.Tab(label='Raw Data', children=[
                html.Div([
                    dcc.Input(
                        id='steps-filter',
                        type='text',
                        placeholder='Filter Training Steps (e.g. 1000 or 500-2000)',
                        style={'width': '300px', 'margin': '10px'}
                    ),
                    html.Button('Apply', id='apply-filter', n_clicks=0, style={'margin': '10px'}),
                    html.Button('Reset', id='reset-filter', n_clicks=0, style={'margin': '10px'})
                ]),
                html.Div(
                    dash_table.DataTable(
                        id='raw-data-table',
                        columns=[{"name": i, "id": i} for i in (df.columns if not df.empty else [])],
                        data=df.to_dict('records') if not df.empty else [],
                        page_size=20,
                        style_table={
                            'overflowX': 'auto',
                            'height': 'calc(100vh - 200px)',
                            'overflowY': 'auto',
                            'flex': '1'
                        },
                        style_cell={
                            'height': 'auto',
                            'minWidth': '100px', 'width': '100px', 'maxWidth': '180px',
                            'whiteSpace': 'normal'
                        },
                        page_current=0,
                        filter_action='native',
                        sort_action='native'
                    ),
                    style={'flex': '1', 'display': 'flex', 'flexDirection': 'column'}
                ),
                dcc.Interval(id='interval-table', interval=1000, n_intervals=0)
            ])
        ]),
        
        dcc.Store(id='graph-state', data={
            'last_data_length': last_data_length,
            'filtered_data': df.to_dict('records') if not df.empty else [],
            'filter_active': False
        })
    ])

    @app.callback(
        [Output('live-graph', 'figure', allow_duplicate=True),
         Output('graph-state', 'data', allow_duplicate=True)],
        [Input('interval-component', 'n_intervals')],
        [State('graph-state', 'data')],
        prevent_initial_call=True
    )
    def update_graph(n, state):
        nonlocal last_data_length
        
        if df.empty or len(df) == state['last_data_length']:
            raise dash.exceptions.PreventUpdate
        
        fig = px.line(df, x='Training Steps', y='Reward', 
                     title='Reward Progress During Training')
        
        fig.update_layout(
            xaxis_title='Training Steps',
            yaxis_title='Reward',
            hovermode='x unified',
            height=600,
            xaxis=dict(rangeslider=dict(visible=True)))
        
        return fig, {
            'last_data_length': len(df),
            'filtered_data': state['filtered_data'],
            'filter_active': state['filter_active']
        }

    @app.callback(
        [Output('components-graph', 'figure', allow_duplicate=True),
         Output('graph-state', 'data', allow_duplicate=True)],
        [Input('interval-components', 'n_intervals')],
        [State('graph-state', 'data')],
        prevent_initial_call=True
    )
    def update_components_graph(n, state):
        nonlocal last_data_length
        
        if df.empty or len(df) == state['last_data_length']:
            raise dash.exceptions.PreventUpdate
        
        reward_components = [col for col in df.columns 
                           if col not in ['Training Steps']
                           and pd.api.types.is_numeric_dtype(df[col])]
        
        fig = go.Figure()
        
        for component in reward_components:
            fig.add_trace(go.Scatter(
                x=df['Training Steps'],
                y=df[component],
                mode='lines',
                name=component
            ))
        
        fig.update_layout(
            title='Reward Components',
            xaxis_title='Training Steps',
            yaxis_title='Value',
            height=600,
            hovermode='x unified',
            xaxis=dict(rangeslider=dict(visible=True)))
        
        return fig, {
            'last_data_length': len(df),
            'filtered_data': state['filtered_data'],
            'filter_active': state['filter_active']
        }

    @app.callback(
        [Output('raw-data-table', 'columns'),
         Output('raw-data-table', 'data'),
         Output('graph-state', 'data'),
         Output('steps-filter', 'value')],
        [Input('interval-table', 'n_intervals'),
         Input('apply-filter', 'n_clicks'),
         Input('reset-filter', 'n_clicks')],
        [State('graph-state', 'data'),
         State('steps-filter', 'value')],
        prevent_initial_call=True
    )
    def update_table(n_intervals, apply_clicks, reset_clicks, state, steps_filter):
        nonlocal df, last_data_length
        
        ctx = dash.callback_context
        if not ctx.triggered:
            trigger_id = None
        else:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Se non ci sono dati, non fare nulla
        if df.empty:
            raise dash.exceptions.PreventUpdate
        
        # Se è stato premuto il pulsante "Reset Filtro"
        if trigger_id == 'reset-filter':
            return (
                [{"name": i, "id": i} for i in df.columns],
                df.to_dict('records'),
                {
                    'last_data_length': len(df),
                    'filtered_data': df.to_dict('records'),
                    'filter_active': False
                },
                ''  # Resetta il valore del campo di input
            )
        
        # Se è stato premuto il pulsante "Applica Filtro"
        if trigger_id == 'apply-filter':
            if steps_filter:
                try:
                    if '-' in steps_filter:
                        # Filtro per range
                        start, end = map(int, steps_filter.split('-'))
                        filtered_df = df[(df['Training Steps'] >= start) & 
                                      (df['Training Steps'] <= end)]
                    else:
                        # Filtro per valore esatto
                        step_value = int(steps_filter)
                        filtered_df = df[df['Training Steps'] == step_value]
                    
                    filtered_data = filtered_df.to_dict('records')
                    return (
                        [{"name": i, "id": i} for i in df.columns],
                        filtered_data,
                        {
                            'last_data_length': len(df),
                            'filtered_data': filtered_data,
                            'filter_active': True
                        },
                        steps_filter  # Mantieni il valore del filtro
                    )
                except ValueError:
                    # Se il filtro non è valido, mostra tutti i dati
                    return (
                        [{"name": i, "id": i} for i in df.columns],
                        df.to_dict('records'),
                        {
                            'last_data_length': len(df),
                            'filtered_data': df.to_dict('records'),
                            'filter_active': False
                        },
                        steps_filter  # Mantieni il valore del filtro
                    )
            else:
                # Se non c'è filtro, mostra tutti i dati
                return (
                    [{"name": i, "id": i} for i in df.columns],
                    df.to_dict('records'),
                    {
                        'last_data_length': len(df),
                        'filtered_data': df.to_dict('records'),
                        'filter_active': False
                    },
                    ''  # Resetta il valore del campo di input
                )
        
        # Se è un aggiornamento automatico
        if len(df) > state['last_data_length']:
            if state['filter_active']:
                # Mantieni il filtro attivo
                filtered_data = state['filtered_data']
                # Aggiungi solo i nuovi dati che corrispondono al filtro
                new_data = df.iloc[state['last_data_length']:].to_dict('records')
                if steps_filter:
                    try:
                        if '-' in steps_filter:
                            start, end = map(int, steps_filter.split('-'))
                            filtered_new_data = [row for row in new_data 
                                              if start <= row['Training Steps'] <= end]
                        else:
                            step_value = int(steps_filter)
                            filtered_new_data = [row for row in new_data 
                                              if row['Training Steps'] == step_value]
                        filtered_data.extend(filtered_new_data)
                    except ValueError:
                        filtered_data = df.to_dict('records')
            else:
                filtered_data = df.to_dict('records')
            
            return (
                [{"name": i, "id": i} for i in df.columns],
                filtered_data,
                {
                    'last_data_length': len(df),
                    'filtered_data': filtered_data,
                    'filter_active': state['filter_active']
                },
                steps_filter  # Mantieni il valore del filtro
            )
        
        # Se non ci sono cambiamenti nei dati
        return (
            [{"name": i, "id": i} for i in df.columns],
            state['filtered_data'] if state['filter_active'] else df.to_dict('records'),
            state,
            steps_filter  # Mantieni il valore del filtro
        )

    app.run(debug=True, port=8050, host='0.0.0.0')

if __name__ == '__main__':
    csv_file_path = '../rewards_continuous.csv'
    launch_dash(csv_file_path)