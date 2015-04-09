
class DynamicAnnotationMixin(object):
    pass


class StaticAnnotationMixin(object):
    pass


class Annotation(object):
    pass


class ImageAnnotation(StaticAnnotationMixin, Annotation):
    pass


class VideoAnnotation(DynamicAnnotationMixin, Annotation):
    pass


class TextAnnotation(DynamicAnnotationMixin, Annotation):
    pass
