from typing import List

from manim import VGroup, Mobject


def mobjects_from_v_group(v_group: VGroup) -> List[Mobject]:
    mobjects: List[Mobject] = []
    for element in v_group.submobjects:
        if isinstance(element, VGroup):
            mobjects.extend(mobjects_from_v_group(element))
        elif isinstance(element, Mobject):
            mobjects.append(element)
        else:
            raise Exception(f'Unexpected element when looking for Mobjects: {element}')
    return mobjects


def v_group_mobject_intersection(v_group0: VGroup, v_group1: VGroup) -> VGroup:
    mobjects0: List[Mobject] = mobjects_from_v_group(v_group0)
    mobjects1: List[Mobject] = mobjects_from_v_group(v_group1)
    return VGroup(*list(set(mobjects0) & set(mobjects1)))


def v_group_mobject_union(v_group0: VGroup, v_group1: VGroup) -> VGroup:
    mobjects0: List[Mobject] = mobjects_from_v_group(v_group0)
    mobjects1: List[Mobject] = mobjects_from_v_group(v_group1)
    return VGroup(*list(set(mobjects0) | set(mobjects1)))


def v_group_mobject_subtraction(v_group0: VGroup, v_group1: VGroup) -> VGroup:
    mobjects0: List[Mobject] = mobjects_from_v_group(v_group0)
    mobjects1: List[Mobject] = mobjects_from_v_group(v_group1)
    return VGroup(*list(set(mobjects0) - set(mobjects1)))
