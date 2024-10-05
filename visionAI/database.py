"""
This file manages the database module of the image annotation tool designed to create training datasets for vision models.

Constraints and Considerations:
1. **Data Structure**:
    - Images: Now include resolution (width and height).
    - Annotations: As previously defined.
2. **Database**:
    - Uses SQLAlchemy ORM for database interactions.
    - Supports multiple SQLite databases (one per dataset).
3. **Functionality**:
    - Ability to add images with resolution and annotations.
    - Retrieve all annotations for a given image.
4. **Flexibility**:
    - DatabaseManager class allows specifying the database file at runtime.
5. **Maintenance and Extensibility**:
    - Well-structured for easy modifications and additions.
6. **Performance**:
    - Efficient session management to prevent memory leaks.

This updated version includes image resolution and maintains the flexibility to manage multiple datasets.
"""

from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, Enum, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import os
import shutil
from PIL import Image as PILImage
from sqlalchemy.exc import IntegrityError

Base = declarative_base()

class Image(Base):
    __tablename__ = 'images'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    path = Column(String, nullable=False, unique=True)
    is_train = Column(Boolean, default=False, nullable=False)
    is_test = Column(Boolean, default=False, nullable=False)
    width = Column(Integer, nullable=False)  # New field for image width
    height = Column(Integer, nullable=False)  # New field for image height
    
    annotations = relationship('Annotation', back_populates='image', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Image(id={self.id}, path='{self.path}', is_train={self.is_train}, is_test={self.is_test}, width={self.width}, height={self.height})>"

class Annotation(Base):
    __tablename__ = 'annotations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    type = Column(Enum('rectangle', 'point', 'line', name='annotation_types'), nullable=False)
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=True)
    y2 = Column(Float, nullable=True)
    class_name = Column(String, nullable=False)
    date_added = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    image = relationship('Image', back_populates='annotations')
    
    def __repr__(self):
        return (
            f"<Annotation(id={self.id}, type='{self.type}', x1={self.x1}, y1={self.y1}, "
            f"x2={self.x2}, y2={self.y2}, class_name='{self.class_name}', "
            f"date_added='{self.date_added}')>"
        )

class DatabaseManager:
    def __init__(self, db_file):
        """
        Initialize the DatabaseManager with a specific database file.
        
        :param db_file: Path to the SQLite database file
        """
        self.db_file = db_file
        self.engine = create_engine(f'sqlite:///{db_file}', echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.init_db()

    def init_db(self):
        """
        Initialize the database by creating all defined tables.
        """
        Base.metadata.create_all(self.engine)

    def add_image(self, path, width, height, is_train=False, is_test=False):
        with self.SessionLocal() as session:
            try:
                # Essayez d'abord d'insérer une nouvelle image
                new_image = Image(path=path, width=width, height=height, is_train=is_train, is_test=is_test)
                session.add(new_image)
                session.commit()
                session.refresh(new_image)
                return new_image
            except IntegrityError:
                # Si l'image existe déjà, faites une mise à jour
                session.rollback()
                existing_image = session.query(Image).filter_by(path=path).first()
                if existing_image:
                    existing_image.width = width
                    existing_image.height = height
                    existing_image.is_train = is_train
                    existing_image.is_test = is_test
                    session.commit()
                    session.refresh(existing_image)
                    return existing_image
                else:
                    print(f"Failed to add or update image: {path}")  # Débogage
                    return None

    def add_annotation(self, image_id, annotation_type, x1, y1, x2, y2, class_name):
        """
        Add a new annotation to the database.
        
        :param image_id: ID of the associated image
        :param annotation_type: Type of annotation (rectangle, point, or line)
        :param x1, y1, x2, y2: Coordinates of the annotation
        :param class_name: Class name of the annotated object
        :return: The created Annotation object
        """
        with self.SessionLocal() as session:
            new_annotation = Annotation(
                image_id=image_id,
                type=annotation_type,
                x1=x1, y1=y1, x2=x2, y2=y2,
                class_name=class_name
            )
            session.add(new_annotation)
            session.commit()
            session.refresh(new_annotation)
            return new_annotation

    def get_image_annotations(self, image_path):
        with self.SessionLocal() as session:
            image = session.query(Image).filter_by(path=image_path).first()
            if image:
                annotations = session.query(Annotation).filter_by(image_id=image.id).all()
                ret = [
                    (ann.type, ann.x1, ann.y1, ann.x2, ann.y2)
                    for ann in annotations
                ]
                print(ret)
                return ret
            print(f"No annotations found for image: {image_path}")
            return []

    def get_candidate_images(self):
        with self.SessionLocal() as session:
            candidates = session.query(Image).filter_by(is_train=False, is_test=False).all()
            print(f"Database query returned {len(candidates)} candidates")  # Débogage
            return candidates

    def get_keyframe_images(self):
        with self.SessionLocal() as session:
            return session.query(Image).filter(Image.is_train | Image.is_test).all()

    def get_image_by_path(self, path):
        with self.SessionLocal() as session:
            # Normalize the path to handle both absolute and relative paths
            normalized_path = os.path.normpath(path)
            return session.query(Image).filter(Image.path.like(f"%{normalized_path}")).first()

    def import_images(self, source_dir, destination_dir):
        imported_count = 0
        for filename in os.listdir(source_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(source_dir, filename)
                dst_path = os.path.join(destination_dir, filename)
                shutil.copy2(src_path, dst_path)
                
                # Add image to database using relative path
                with PILImage.open(dst_path) as img:
                    width, height = img.size
                relative_path = os.path.relpath(dst_path, os.path.dirname(self.db_file))
                new_image = self.add_image(relative_path, width, height)
                print(f"Added image to database: {new_image}")  # Débogage
                
                imported_count += 1
        print(f"Imported {imported_count} images")  # Débogage
        return imported_count

    def save_annotations(self, image_path, annotations):
        with self.SessionLocal() as session:
            image = session.query(Image).filter_by(path=image_path).first()
            if image:
                try:
                    # Remove existing annotations
                    session.query(Annotation).filter_by(image_id=image.id).delete()
                    
                    # Add new annotations
                    for ann_type, x1, y1, x2, y2 in annotations:
                        new_annotation = Annotation(
                            image_id=image.id,
                            type=ann_type,
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            class_name="default"
                        )
                        session.add(new_annotation)
                    
                    session.commit()
                    print(f"Saved {len(annotations)} annotations for {image_path}")
                except Exception as e:
                    session.rollback()
                    print(f"Error saving annotations: {e}")
            else:
                print(f"Image not found in database: {image_path}")

    @staticmethod
    def create_database(db_path):
        engine = create_engine(f'sqlite:///{db_path}', echo=False)
        Base.metadata.create_all(engine)

    def close(self):
        self.engine.dispose()

# Usage example:
if __name__ == "__main__":
    db_manager = DatabaseManager("dataset1.db")
    
    # Add an image with resolution
    image = db_manager.add_image("/path/to/image.jpg", width=1920, height=1080, is_train=True)
    
    # Add an annotation
    annotation = db_manager.add_annotation(
        image_id=image.id,
        annotation_type="rectangle",
        x1=10, y1=20, x2=100, y2=200,
        class_name="car"
    )
    
    # Get annotations for the image
    annotations = db_manager.get_image_annotations(image.id)
    for ann in annotations:
        print(ann)