from keras import layers, models, optimizers, regularizers
from keras import backend as K
import numpy as np

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, params={}):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.dropout_rate = 0.4
        self.batch_norm = params.get("batch_norm", False)
        self.size_multiplicator = 2
        if(params.get("size_multiplicator")):
            self.size_multiplicator = params.get("size_multiplicator")
        if(params.get("dropout_rate")):
            self.dropout_rate = params.get("dropout_rate")
        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=self.size_multiplicator*16, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.01))(states)
        net_states = layers.Dropout(self.dropout_rate)(net_states)
        if self.batch_norm:
            net = layers.BatchNormalization()(net)
        net_states = layers.Dense(units=self.size_multiplicator*32, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.01))(net_states)
        net_states = layers.Dropout(self.dropout_rate)(net_states)
        if self.batch_norm:
            net = layers.BatchNormalization()(net)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=self.size_multiplicator*16, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.01))(actions)
        net_actions = layers.Dropout(self.dropout_rate)(net_actions)
        if self.batch_norm:
            net = layers.BatchNormalization()(net)
        net_actions = layers.Dense(units=self.size_multiplicator*32, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.01))(net_actions)
        net_actions = layers.Dropout(self.dropout_rate)(net_actions)
        if self.batch_norm:
            net = layers.BatchNormalization()(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        # net = layers.Activation('relu')(net)
        # net = layers.Dense(units=size_multiplicator_merge*8, kernel_initializer='random_uniform', activation='relu', kernel_regularizer=regularizers.l2(0.01))(net)
        # net = layers.Dropout(self.dropout_rate)(net)
        # net = layers.Dense(units=size_multiplicator_merge*4, kernel_initializer='random_uniform', activation='relu', kernel_regularizer=regularizers.l2(0.01))(net)
        # net = layers.Dropout(self.dropout_rate)(net)
        # net = layers.Dense(units=size_multiplicator*32, kernel_initializer='random_uniform', activation='relu', kernel_regularizer=regularizers.l2(0.01))(net)
        # net = layers.Dropout(self.dropout_rate)(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values', kernel_initializer='random_uniform')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
