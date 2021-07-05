#%%
from tensorflow import keras

# Define the Required Callback Function
class printlearningrate(keras.callbacks.Callback):
	"""
	Keras custom callback to print the learning rate at the end of each epoch
	"""
	def on_epoch_end(self, epoch, logs={}):
		optimizer = self.model.optimizer
		lr = keras.backend.eval(optimizer.lr)
		Epoch_count = epoch + 1; print('\n', "Epoch:", Epoch_count, ', LR: {:.6f}'.format(lr))


class BananaDetectionModel():

	def CNN_hidden(self, image_input):
		"""
		Common hidden CNN module of the network
		"""
		# CNN layers
		X = keras.layers.Conv2D(8, (3,3), strides=(1,1), padding='same')(image_input)
		X = keras.layers.Activation('relu')(X)
		X = keras.layers.BatchNormalization(axis=-1)(X)
		X = keras.layers.MaxPooling2D(pool_size=(2,2))(X)

		X = keras.layers.Conv2D(16, (3,3), strides=(1,1), padding='same')(X)
		X = keras.layers.Activation('relu')(X)
		X = keras.layers.BatchNormalization(axis=-1)(X)
		X = keras.layers.MaxPooling2D(pool_size=(2,2))(X)
		X = keras.layers.Dropout(0.2, seed=123)(X)

		X = keras.layers.Conv2D(32, (3,3), strides=(1,1), padding='same')(X)
		X = keras.layers.Activation('relu')(X)
		X = keras.layers.BatchNormalization(axis=-1)(X)
		X = keras.layers.MaxPooling2D(pool_size=(2,2))(X)

		return X

	def Classifier_branch(self, image_input):
		"""
		Classifier branch of the network
		"""

		X = self.CNN_hidden(image_input)
		# Dense layers
		X = keras.layers.Flatten()(X)
		X = keras.layers.Dropout(0.2, seed=123)(X)
		X = keras.layers.Dense(10)(X)
		X = keras.layers.Activation('relu')(X)
		X = keras.layers.Dense(1)(X)
		X = keras.layers.Activation('sigmoid', name = 'classifier_output')(X)

		return X


	def Regression_branch(self, image_input):
		"""
		Regressor branch of the network
		"""
		X = self.CNN_hidden(image_input)
		# Dense layers
		X = keras.layers.Flatten()(X)
		X = keras.layers.Dropout(0.2, seed=123)(X)
		X = keras.layers.Dense(10)(X)
		X = keras.layers.Activation('relu')(X)
		X = keras.layers.Dense(1)(X)
		X = keras.layers.Activation('sigmoid', name = 'regression_output')(X)

		return X

	def Classifier(self, input_shape):
		"""
		Model builder for simple classification purposes
		"""
		image_input = keras.Input(input_shape, name='image')

		class_branch = self.Classifier_branch(image_input)

		model = keras.Model(inputs = image_input, 
							outputs = class_branch, 
							name = 'Banana_classifier')
		return model

	def Classifier_with_regression(self, input_shape):
		"""
		Model builder to jointly train classification and regression task
		"""

		image_input = keras.Input(input_shape, name='image')

		class_branch = self.Classifier_branch(image_input)
		reg_branch = self.Regression_branch(image_input)

		model = keras.Model(inputs = image_input, 
							outputs = [class_branch, reg_branch], 
							name = 'Banana_classregressor')
		return model


if __name__ == '__main__':

	shape = [50, 50, 3]

	class_model_new = BananaDetectionModel().Classifier(shape)
	class_model_new.summary()

	classreg_model = BananaDetectionModel().Classifier_with_regression(shape)
	classreg_model.summary()

	keras.utils.plot_model(classreg_model, to_file='reg_model.png', show_shapes=True, show_layer_names=True, rankdir='TB')
