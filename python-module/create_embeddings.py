import numpy as np

with open('embeddings.txt', 'wb') as f:
    np.savetxt(f, np.random.rand(10001, 512), delimiter=',', fmt='%1.4e')

with open('fastapi_test/embeddings.txt', 'wb') as f:
    np.savetxt(f, np.random.rand(10001, 512), delimiter=',', fmt='%1.4e')

with open('flask_test/embeddings.txt', 'wb') as f:
    np.savetxt(f, np.random.rand(10001, 512), delimiter=',', fmt='%1.4e')



