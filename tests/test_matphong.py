from matPhong import MatPhong


def test_matphong_initialization() -> None:
    ambient_coefficient: float = 0.2
    diffuse_coefficient: float = 0.5
    specular_coefficient: float = 0.8
    shininess_exponent: float = 32.0

    material: MatPhong = MatPhong(
        ka=ambient_coefficient,
        kd=diffuse_coefficient,
        ks=specular_coefficient,
        n=shininess_exponent,
    )

    assert material.ka == ambient_coefficient
    assert material.kd == diffuse_coefficient
    assert material.ks == specular_coefficient
    assert material.n == shininess_exponent
