import numpy as np

#Performs linear LSR to fit a model based on inputs
def multi_regress(y,z):
    """Perform multiple linear regression.

    Parameters
    ----------
    y : array_like, shape = (n,1)
        The vector of dependent variable data
    Z : array_like, shape = (n,m)
        The matrix of independent variable data

    Returns
    -------
    a: numpy.ndarray, shape = (m,1)
        The vector of model coefficients
    e: numpy.ndarray, shape = (n,1)
        The vector of residuals
    rsq: float
        R^2 value of model, coefficient of determination
    model: numpy.ndarray, shape = (m,1)
        Vector of model output
"""

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