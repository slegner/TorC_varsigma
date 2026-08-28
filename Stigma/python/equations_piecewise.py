
def cp_plus(stigma, wi):
    val = (-2*((-1 + stigma)**2*stigma**2*(2 - 6*wi + (stigma + \
         3*wi*stigma)**2))**(1/2) + stigma*(1 + 2*(-3 + stigma)*stigma + \
         wi*(-3 + 6*stigma*(1 + stigma))))/((-1 + 3*wi)*(-1 + 4*stigma**2))
    return val

def cp_minus(stigma, wi):
    val = (2*((-1 + stigma)**2*stigma**2*(2 - 6*wi + (stigma + \
         3*wi*stigma)**2))**(1/2) + stigma*(1 + 2*(-3 + stigma)*stigma + \
         wi*(-3 + 6*stigma*(1 + stigma))))/((-1 + 3*wi)*(-1 + 4*stigma**2))
    return val
