import jax
import jax.numpy as jnp
import time

print(jnp.array(1).device().device_kind)

@jax.jit
def f(x, y):
  return jnp.einsum('bqc,bkc->bqk', x, y)

x_bfloat = jnp.ones((384 * 4, 384, 16), dtype=jnp.bfloat16)
x_float = jnp.ones((384 * 4, 384, 16), dtype=jnp.float16)

# Warmup
_ = f(x_bfloat, x_bfloat)
_ = f(x_float, x_float)

start_time = time.time()
result = f(x_float, x_float).block_until_ready()
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: {:.3f} seconds".format(elapsed_time))

start_time = time.time()
result = f(x_bfloat, x_bfloat).block_until_ready()
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: {:.3f} seconds".format(elapsed_time))

#%timeit f(x_float, x_float).block_until_ready()
#%timeit f(x_bfloat, x_bfloat).block_until_ready()
