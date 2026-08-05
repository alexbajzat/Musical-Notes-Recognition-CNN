

# Reconocimiento-De-Notas-Musicales-CNN
Red neuronal convolucional para el reconocimiento de notas musicales desde partituras.

# Mejores configuraciones hasta ahora
STEP_SIZE = 1e-6
FEATURE_STEP_SIZE = 1e-6
REG = 1e-4
BATCH_SIZE = 32
FULLY_CONNECTED_NEURONS = 50
ITERATIONS = 50

conv(5 filters, 3X3) => relu => conv(5 filters, 3X3) => relu => pool => conv(15 filters, 3X3) => relu => pool => flat => 2*hidden => softmax



Intervalo/Paso de confianza
Análisis estadístico
