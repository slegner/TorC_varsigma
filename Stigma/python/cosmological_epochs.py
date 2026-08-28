def HubbleRadiation(OmegaR, wi, stigma, ups2, kappa=1, H0=1):
    """Modified Hubble parameter for radiation-dominated epoch"""
    return (2*H0*(OmegaR)**(1/2)*((kappa*(stigma*(3 - 9*wi) - stigma**3*(1 + 3*wi)**2 + (stigma + 3*stigma*wi)**2 - (1 + 3*wi)*((-1 + stigma)**2*stigma**2*(2 - 6*wi + (stigma + 3*stigma*wi)**2))**(1/2)))/(stigma*ups2*(1 + 2*stigma + 6*(-1 + stigma)*wi)*(-7 - 3*wi + 4*stigma*(1 + 3*wi)))))/(3)**(1/2)

def HubbleMatter(OmegaM, wi, stigma, ups2, kappa=1, H0=1):
    """Modified Hubble parameter for matter-dominated epoch"""
    return (2*H0*(OmegaM)**(1/2)*((kappa*(stigma*(3 - 9*wi) - stigma**3*(1 + 3*wi)**2 + (stigma + 3*stigma*wi)**2 - (1 + 3*wi)*((-1 + stigma)**2*stigma**2*(2 - 6*wi + (stigma + 3*stigma*wi)**2))**(1/2)))/(stigma*ups2*(1 + 2*stigma + 6*(-1 + stigma)*wi)*(-7 - 3*wi + 4*stigma*(1 + 3*wi)))))/(3)**(1/2)

def HubbleLambda(OmegaL, wi, stigma, ups2, kappa=1, H0=1):
    """Modified Hubble parameter for Lambda-dominated epoch"""
    return (2*H0*(OmegaL)**(1/2)*((kappa*(stigma*(3 - 9*wi) - stigma**3*(1 + 3*wi)**2 + (stigma + 3*stigma*wi)**2 - (1 + 3*wi)*((-1 + stigma)**2*stigma**2*(2 - 6*wi + (stigma + 3*stigma*wi)**2))**(1/2)))/(stigma*ups2*(1 + 2*stigma + 6*(-1 + stigma)*wi)*(-7 - 3*wi + 4*stigma*(1 + 3*wi)))))/(3)**(1/2)

def HubbleTotal(OmegaR, OmegaM, OmegaL, a, wi, stigma, ups2, kappa=1, H0=1):
    """Modified Hubble parameter for all epochs combined: H = H0 * sqrt(gi * (OmegaR/a^4 + OmegaM/a^3 + OmegaL))"""
    return (2*H0*(OmegaL + OmegaM/a**3 + OmegaR/a**4)**(1/2)*((kappa*(stigma*(3 - 9*wi) - stigma**3*(1 + 3*wi)**2 + (stigma + 3*stigma*wi)**2 - (1 + 3*wi)*((-1 + stigma)**2*stigma**2*(2 - 6*wi + (stigma + 3*stigma*wi)**2))**(1/2)))/(stigma*ups2*(1 + 2*stigma + 6*(-1 + stigma)*wi)*(-7 - 3*wi + 4*stigma*(1 + 3*wi)))))/(3)**(1/2)