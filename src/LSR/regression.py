import numpy as np

def multi_regress(y,z):

    ztz = np.matmul(np.transpose(z),z)
    zty = np.matmul(np.transpose(z),y)
    a = np.matmul(np.linalg.inv(ztz),zty)
    m_ave = np.full(y.shape,np.mean(y))
    e_ave = y - m_ave
    ebte = float(np.matmul(np.transpose(e_ave),e_ave))
    m_line = np.matmul(z,a)
    e_line = y - m_line
    ete = float(np.matmul(np.transpose(e_line),e_line))
    rsq = (ebte - ete) / ebte

    return a,e_line,rsq, m_line