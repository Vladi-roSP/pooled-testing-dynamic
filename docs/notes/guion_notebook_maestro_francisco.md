# Guion para presentar el notebook maestro a Francisco

Notebook base: `augmented/notebooks/notebook_maestro.ipynb`

La idea de este guion es tener una forma clara y natural de explicar cada gráfica. No es para leerlo como paper. Es para usarlo como apoyo en la conversación con Francisco: qué está pasando, por qué importa y qué pregunta puede abrir un insight.

## Mensaje general

Lo que quiero comunicar con el notebook es esto:

Dynamic Augmented Pooled Testing no es solamente pooled testing con tests más informativos. El conteo exacto cambia la forma en que aprendemos. Cada test no solo actualiza probabilidades individuales; también crea relaciones entre personas, cambia las decisiones futuras y puede volver difícil calcular bien la posterior.

El hilo de las gráficas es:

1. Primero mostramos que el óptimo puede separarse bastante del greedy incluso en una instancia mínima.
2. Luego mostramos por qué: los conteos exactos generan información cruzada.
3. Después mostramos que esa información crea dependencias, como super-nodos.
4. Luego aclaramos cómo entra la utilidad: solo ganamos cuando un pool sale completamente negativo.
5. Después mostramos un problema para muestreo: la fibra puede partirse.
6. Cerramos con la razón computacional: la dificultad depende de cómo se cruzan los pools.

## Gráfica 1: Greedy vs óptimo

Esta gráfica es interesante porque es el ejemplo más chico que encontramos donde la separación entre greedy y óptimo ya es significativa.

La instancia tiene:

```text
n = 4
B = 2
G = 3
```

El greedy obtiene:

```text
U_greedy = 3.700
```

El óptimo obtiene:

```text
U_opt = 4.187
```

La diferencia es:

```text
gap = 11.6%
```

Lo importante no es solo que el óptimo gane. Lo importante es cómo gana.

El greedy empieza testeando una sola persona, `{1}`. Esa decisión tiene sentido si uno piensa solo en el beneficio inmediato. Parece una jugada limpia: testeo a alguien, si sale sano lo libero, y sigo.

El óptimo hace algo menos obvio: empieza con `{1,2}`. Ese pool puede devolver 0, 1 o 2 infectados. Cada resultado abre una rama distinta para el segundo test. Entonces el primer test no se valora solo por lo que libera hoy, sino por la información que deja para decidir mañana.

La frase clave sería:

El greedy elige el pool que se ve mejor ahora. El óptimo elige el pool que deja mejor parado al árbol completo.

Este ejemplo sirve para justificar por qué el problema es realmente dinámico. Si hubiera que resumirlo en una línea: el valor del primer test está en cómo organiza el segundo.

Preguntas para Francisco:

1. ¿Qué está viendo el óptimo en el primer pool `{1,2}` que el greedy no alcanza a valorar?
2. ¿Este gap de 11.6% te parece una excepción de esta instancia o una señal de que el greedy falla estructuralmente en el problema dinámico?

## Gráfica 2: Información cruzada

Esta gráfica es importante porque muestra, con el ejemplo más simple posible, que dos tests juntos pueden decir algo que cada test por separado no deja ver.

Tenemos:

```text
t1 = {0,1}, conteo = 1
t2 = {1,2}, conteo = 0
```

El segundo test dice que en `{1,2}` hay cero infectados. Entonces 1 y 2 están sanos.

Pero el primer test decía que en `{0,1}` había exactamente un infectado. Como ya sabemos que 1 está sano, el infectado tiene que ser 0.

Por eso la actualización correcta da:

```text
P(0 infectado | historia completa) = 1.00
```

La actualización local deja a 0 en:

```text
P(0 infectado) = 0.30
```

porque no cruza bien la información entre los dos tests.

El punto no es que hacer Bayes secuencial sea malo. El punto es que si en cada paso solo guardamos marginales individuales, podemos perder relaciones importantes. Aquí la relación es muy clara: un test vuelve sanos a 1 y 2, y eso obliga al otro test a señalar a 0.

La frase clave sería:

La información no está en cada test aislado; está en cómo se cruzan los tests.

Esta gráfica justifica por qué necesitamos pensar en la historia completa y no solo en actualizaciones locales.

Preguntas para Francisco:

1. ¿Qué información estamos perdiendo cuando actualizamos solo con marginales individuales?
2. ¿Este ejemplo te parece suficiente para justificar que la posterior debe calcularse usando toda la historia de tests junta?

## Gráfica 3: Super-nodo

Esta gráfica es interesante porque muestra qué pasa después de observar un conteo exacto dentro de un pool pequeño.

Se testea:

```text
{3,6}
```

y el resultado es:

```text
conteo = 1
```

Eso significa:

```text
exactamente uno de los dos está infectado
```

Antes del test, podíamos pensar en 3 y 6 como dos personas separadas. Después del test, ya no. Si 3 está infectado, 6 tiene que estar sano. Si 6 está infectado, 3 tiene que estar sano.

Por eso aparece el super-nodo `S = {3,6}`. No significa que 3 y 6 sean la misma persona. Significa que ahora forman una unidad probabilística. El estado de uno afecta directamente lo que puede pasar con el otro.

Los números ayudan a verlo:

```text
prior p3 = 0.24
prior p6 = 0.15
posterior q3 = 0.642
posterior q6 = 0.358
q3 + q6 = 1.000
```

No quedan 0.5 y 0.5 porque antes del test 3 ya era más riesgoso que 6. Condicionado a que exactamente uno está infectado, es más probable que sea 3.

Pero lo más importante es esto: no podemos tomar `q3 = 0.642` y `q6 = 0.358` como si fueran independientes. La probabilidad de que ambos estén infectados no es `0.642 * 0.358`. Es cero, porque el conteo dijo que exactamente uno está infectado.

La frase clave sería:

El super-nodo aparece porque el conteo no solo cambia probabilidades; crea una dependencia.

Esta gráfica sirve para explicar por qué augmented pooled testing necesita guardar más estructura que solo probabilidades individuales.

Preguntas para Francisco:

1. ¿Qué tenemos que guardar realmente sobre `{3,6}` después del conteo: solo sus probabilidades individuales o la distribución conjunta entre ellos?
2. ¿Cuándo este tipo de super-nodo deja de ser una simplificación útil y empieza a volverse un problema computacional?

## Gráfica 4: Utilidad y pools completamente negativos

Esta gráfica aclara algo básico del objetivo: la utilidad solo se gana cuando el pool sale completamente negativo.

Si un pool tiene conteo 0, todos sus miembros están sanos y se pueden liberar. Ahí se gana la suma de sus utilidades.

La probabilidad de que eso pase es:

```text
P(todo sano) = producto de q_i
```

donde `q_i` es la probabilidad de que la persona `i` esté sana.

Entonces el valor aproximado del pool es:

```text
valor = P(todo sano) * suma de utilidades
```

La gráfica muestra el tradeoff. Meter más personas puede subir la utilidad potencial, pero también baja la probabilidad de que todo el pool salga sano.

El ejemplo más claro es la persona E. E tiene utilidad alta, pero también es riesgosa. Al meterla, la probabilidad de que todo el pool sea sano cae mucho.

Números de la gráfica:

```text
pool {A,B,C,D}:   P(todo sano) = 0.252, valor = 2.52
pool {A,B,C,D,E}: P(todo sano) = 0.076, valor = 1.06
mejor pool {A,B,C}: P(todo sano) = 0.504, valor = 3.53
```

La lectura es simple: un pool más grande no siempre es mejor. Si metemos a alguien muy riesgoso, puede arruinar la probabilidad de limpiar a todos.

La frase clave sería:

El algoritmo no busca pools grandes; busca pools con buen balance entre utilidad y probabilidad de salir negativos.

Esta gráfica ayuda a separar dos ideas: utilidad potencial y probabilidad de cobrar esa utilidad.

Preguntas para Francisco:

1. ¿Qué pesa más en la decisión de un pool: sumar utilidad alta o proteger la probabilidad de que salga completamente negativo?
2. ¿Esta forma de valorar pools captura bien el objetivo real, o también deberíamos valorar información aunque el pool no salga negativo?

## Gráfica 5: La fibra de perfiles válidos

Esta gráfica es útil porque traduce un problema de muestreo a una imagen muy concreta.

La fibra es el conjunto de mundos posibles que todavía son compatibles con los conteos observados.

En el ejemplo tenemos dos restricciones:

```text
{0,1,2} tiene 1 infectado
{2,3,4} tiene 1 infectado
```

Eso deja cinco perfiles válidos:

```text
{2}
{0,3}
{0,4}
{1,3}
{1,4}
```

Pero esos perfiles se separan en dos grupos:

```text
{2} tiene 1 infectado total
los otros cuatro tienen 2 infectados totales
```

El movimiento tipo Gibbs que se ilustra intercambia un sano por un infectado. Ese movimiento mantiene fijo el total de infectados. Entonces puede moverse dentro del grupo rojo, pero no puede cruzar hacia el perfil azul `{2}`.

El problema es sutil: el sampler puede estar haciendo movimientos válidos, pero aun así no recorrer todo el espacio.

La frase clave sería:

No basta con generar perfiles válidos; el método también tiene que poder llegar a todos los perfiles válidos.

Esta gráfica justifica tener cuidado con Gibbs o con cualquier sampler local. Si la fibra se parte, una cadena puede quedarse encerrada en una componente y estimar mal las probabilidades.

Preguntas para Francisco:

1. ¿Qué sampler se quedaría atrapado en una sola componente de esta fibra?
2. ¿Cómo podemos asegurarnos de que el método de muestreo sí recorra todos los perfiles compatibles con los conteos?

## Gráfica 6: Tratabilidad

Esta gráfica resume por qué algunas instancias son fáciles y otras se vuelven caras.

La diferencia no está solo en cuántas personas hay. Está en cómo se cruzan los pools.

Hay cuatro casos:

```text
pools separados
pools anidados
pools en cadena
pools muy cruzados
```

Los tres primeros son relativamente rápidos porque tienen estructura. Se pueden partir, resolver de adentro hacia afuera, o avanzar por una cadena.

El caso difícil es el de pools muy cruzados. Ahí todos los tests comparten información con todos. No hay una forma limpia de separar el problema, entonces calcular la posterior exacta puede requerir mirar muchísimas combinaciones.

La forma simple de decirlo:

Si los pools se cruzan poco, la inferencia tiene atajos. Si se cruzan demasiado, los atajos desaparecen.

Esta gráfica conecta el diseño de pools con el costo computacional. Un pool puede ser informativo, pero también puede hacer que la posterior sea más difícil de calcular.

La frase clave sería:

La geometría de los pools decide si el posterior se calcula rápido o si explota combinatoriamente.

Preguntas para Francisco:

1. ¿Qué patrón de solapamiento entre pools es el que realmente vuelve difícil la inferencia?
2. ¿Deberíamos diseñar los pools pensando también en que la posterior sea tratable, no solo en que el test sea informativo?

## Cierre

Yo cerraría la presentación así:

Estas gráficas muestran que el problema augmented tiene tres capas. Primero está la capa de decisión: el óptimo puede ganarle al greedy porque piensa en el futuro. Luego está la capa estadística: los conteos exactos crean información cruzada y dependencias. Finalmente está la capa computacional: calcular o muestrear bien la posterior puede ser difícil dependiendo de cómo se cruzan los pools.

Entonces el reto no es solamente encontrar pools con alta utilidad esperada. El reto es escoger tests que den buena información, permitan buenas decisiones futuras y mantengan la inferencia manejable.

Pregunta final para Francisco:

Si tuviéramos que vender una contribución central del proyecto, ¿cuál sería: el valor dinámico frente al greedy, la inferencia conjunta con conteos exactos, o el manejo computacional de posteriors complejas?
